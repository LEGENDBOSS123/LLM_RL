"""TUI chat for local MLX models:  python chat.py   (opens the model picker)
                                   python chat.py <model_dir> [--r1zero]   (skip the picker)

Model picker: choose a model from models/, the prompt format and generation settings.
Chat: everything is shown verbatim - system prompt / template, user text, and every character the
model generates (think/answer tags, special tokens).
Commands: /back (model picker)  /system <text>  /nosystem  /temp <float>  /max <int>
          /raw (toggle full prompt)  /clear  /quit
"""
import json, sys, pathlib
from rich.text import Text
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.widgets import Input, Static, Header, Footer, DataTable, RadioSet, RadioButton, Button, Label
from textual import work

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import SYSTEM

MODELS = pathlib.Path(__file__).parent / "models"
HELP = __doc__
R1Z = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
       "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
       "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>.")

NOTES = {
    "Qwen3-1.7B-RL": "★ our zero-RL model (step 180): math 64%, unseen 39%, code 42%",
    "Qwen3-1.7B-Base": "raw pretrained base - the starting point of the RL run",
    "Qwen3-1.7B-Instruct": "official instruct/thinking model (collapsed diversity)",
    "Qwen2.5-1.5B-sft": "SFT'd to think without a system prompt",
    "Qwen2.5-1.5B-tools": "tool-SFT model (collapsed: always runs code)",
    "Qwen2.5-1.5B-Instruct": "original instruct model",
    "Qwen2.5-0.5B-Instruct": "tiny instruct model",
    "Qwen2.5-3B-Instruct": "3B instruct model (slowest, strongest Qwen2.5)",
}


def T(label, body, style):
    """Label in color + body as plain literal text (no markup parsing)."""
    t = Text()
    t.append(label, style=style)
    t.append(body)
    return t


def model_info(path: pathlib.Path):
    cfg = {}
    try:
        cfg = json.load(open(path / "config.json"))
    except Exception:
        pass
    size = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9
    quant = cfg.get("quantization", {}).get("bits") if isinstance(cfg.get("quantization"), dict) else None
    name = path.name
    base_like = ("Base" in name) or name.startswith("Qwen3-1.7B-RL")
    fmt = "r1zero" if base_like else "chat"
    note = next((v for k, v in NOTES.items() if name.startswith(k)), "")
    return dict(name=name, size=size, quant=f"{quant}-bit" if quant else "bf16", fmt=fmt, note=note,
                arch=cfg.get("model_type", "?"))


# ------------------------------------------------------------------ picker
class PickerScreen(Screen):
    BINDINGS = [("enter", "start", "Start chat"), ("q", "app.quit", "Quit")]
    CSS = """
    #title { padding: 1 2 0 2; text-style: bold; }
    #sub { padding: 0 2 1 2; color: $text-muted; }
    DataTable { height: 1fr; margin: 0 2; }
    #opts { height: auto; margin: 1 2; }
    #opts Vertical { width: 1fr; height: auto; padding: 0 1; }
    #opts Input { width: 16; }
    #go { margin: 0 2 1 2; width: 100%; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Choose a model", id="title")
        yield Label("↑/↓ to select, Enter to start. The prompt format is auto-picked per model (base/RL models use the "
                    "plain R1-Zero template; instruct/SFT models use their chat template).", id="sub")
        yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
        with Horizontal(id="opts"):
            with Vertical():
                yield Label("Prompt format")
                with RadioSet(id="fmt"):
                    yield RadioButton("Auto (recommended)", value=True, id="fmt_auto")
                    yield RadioButton("Chat template + think system prompt", id="fmt_chat")
                    yield RadioButton("Chat template, no system prompt", id="fmt_nosys")
                    yield RadioButton("R1-Zero plain text (base / RL models)", id="fmt_r1z")
            with Vertical():
                yield Label("Temperature")
                yield Input("0.7", id="temp")
                yield Label("Max tokens")
                yield Input("1024", id="max")
        yield Button("Start chat ▶", variant="primary", id="go")
        yield Footer()

    def on_mount(self):
        t = self.query_one("#models", DataTable)
        t.add_columns("Model", "Precision", "Size", "Arch", "Format", "Notes")
        self.infos = [model_info(p) for p in sorted(MODELS.iterdir()) if p.is_dir() and (p / "config.json").exists()]
        # our trained model first
        self.infos.sort(key=lambda i: (not i["name"].startswith("Qwen3-1.7B-RL"), i["name"]))
        for i in self.infos:
            t.add_row(i["name"], i["quant"], f"{i['size']:.1f} GB", i["arch"], i["fmt"], i["note"])
        t.focus()

    def on_data_table_row_selected(self, ev: DataTable.RowSelected):
        self.action_start()

    def on_button_pressed(self, ev: Button.Pressed):
        if ev.button.id == "go":
            self.action_start()

    def action_start(self):
        t = self.query_one("#models", DataTable)
        info = self.infos[t.cursor_row]
        choice = self.query_one("#fmt", RadioSet).pressed_button
        cid = choice.id if choice else "fmt_auto"
        if cid == "fmt_auto":
            fmt, system = info["fmt"], ("" if info["fmt"] == "r1zero" else SYSTEM)
        elif cid == "fmt_chat":
            fmt, system = "chat", SYSTEM
        elif cid == "fmt_nosys":
            fmt, system = "chat", ""
        else:
            fmt, system = "r1zero", ""
        try:
            temp = float(self.query_one("#temp", Input).value)
            max_tokens = int(self.query_one("#max", Input).value)
        except ValueError:
            self.notify("temperature / max tokens must be numbers", severity="error")
            return
        self.app.push_screen(ChatScreen(info["name"], fmt, system, temp, max_tokens))


# ------------------------------------------------------------------ chat
class ChatScreen(Screen):
    BINDINGS = [("ctrl+l", "clear", "Clear"), ("ctrl+b", "back", "Models"), ("ctrl+c", "app.quit", "Quit")]
    CSS = """
    #log { height: 1fr; border: round $primary; padding: 0 1; }
    #log Static { margin-bottom: 1; }
    #status { height: 1; color: $text-muted; padding: 0 1; }
    """

    def __init__(self, model_name, fmt, system, temp, max_tokens):
        super().__init__()
        self.model_name, self.template = model_name, ("r1zero" if fmt == "r1zero" else "")
        self.system, self.temp, self.max_tokens = system, temp, max_tokens
        self.model = self.tok = None
        self.history = []
        self.show_raw = False
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Static("", id="status")
        yield Input(placeholder="message  (/help, /back for model picker)", id="inp")
        yield Footer()

    def on_mount(self):
        self.log_w = self.query_one("#log", VerticalScroll)
        self.status = self.query_one("#status", Static)
        self.query_one("#inp").focus()
        self.load_model(self.model_name)

    def add(self, renderable) -> Static:
        w = Static(renderable if not isinstance(renderable, str) else Text(renderable))
        self.log_w.mount(w)
        self.log_w.scroll_end(animate=False)
        return w

    def note(self, s, style="dim"):
        self.add(Text(s, style=style))

    def show_system(self):
        if self.tok is None:
            return
        if self.template:
            self.add(T("prompt template (R1-Zero, base/RL models)> ", R1Z + " User: <your message> Assistant:", "bold magenta"))
        elif self.system:
            self.add(T("system> ", self.system, "bold magenta"))
        else:
            probe = self.tok.apply_chat_template([{"role": "user", "content": "X"}], add_generation_prompt=False, tokenize=False)
            self.add(T("system> (none set — template injects defaults, raw prompt:)\n", probe, "bold magenta"))

    @work(thread=True, exclusive=True, group="load")
    def load_model(self, name):
        from mlx_lm import load
        self.busy = True
        self.app.call_from_thread(self.status.update, f"loading {name}...")
        try:
            self.model, self.tok = load(str(MODELS / name))
            self.history = []
            self.app.call_from_thread(self.note, f"loaded {name}  (format: {'R1-Zero plain text' if self.template else 'chat template'})  /help for commands", "green")
            self.app.call_from_thread(self.show_system)
            self.app.call_from_thread(self.update_status)
        except Exception as e:
            self.app.call_from_thread(self.note, f"load failed: {e}", "red")
        self.busy = False

    def update_status(self, extra=""):
        self.status.update(f"{self.model_name} | {'r1zero' if self.template else 'chat'} | temp {self.temp} | "
                           f"max {self.max_tokens} | raw {self.show_raw} {extra}")

    def action_clear(self):
        self.history = []
        self.log_w.remove_children()
        self.show_system()

    def action_back(self):
        self.model = self.tok = None
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        self.app.pop_screen()

    def on_input_submitted(self, ev: Input.Submitted):
        text = ev.value
        ev.input.value = ""
        if not text.strip():
            return
        if text.startswith("/"):
            return self.command(text.strip())
        if self.busy or self.model is None:
            return self.note("busy, wait...", "yellow")
        self.add(T("user> ", text, "bold cyan"))
        self.generate(text)

    def command(self, text):
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()
        if cmd == "/quit":
            self.app.exit()
        elif cmd == "/back":
            self.action_back()
        elif cmd == "/clear":
            self.action_clear()
        elif cmd == "/system":
            self.system = arg
            self.history = []
            self.note("system prompt set, history cleared")
            self.show_system()
        elif cmd == "/nosystem":
            self.system = ""
            self.history = []
            self.show_system()
        elif cmd == "/temp":
            self.temp = float(arg or 0.7)
            self.update_status()
        elif cmd == "/max":
            self.max_tokens = int(arg or 1024)
            self.update_status()
        elif cmd == "/raw":
            self.show_raw = not self.show_raw
            self.update_status()
        else:
            self.add(Text(HELP))

    @work(thread=True, exclusive=True, group="gen")
    def generate(self, user_text):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.busy = True
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + self.history
        msgs.append({"role": "user", "content": user_text})
        if self.template:
            prompt = R1Z
            for mm in msgs:
                if mm["role"] == "user":
                    prompt += " User: " + mm["content"]
                elif mm["role"] == "assistant":
                    prompt += " Assistant: " + mm["content"]
            prompt += " Assistant:"
        else:
            prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if self.show_raw:
            self.app.call_from_thread(self.add, T("raw prompt>\n", prompt, "bold yellow"))
        sampler = make_sampler(temp=self.temp, top_p=0.95)
        label = f"{self.model_name}> "
        w = self.app.call_from_thread(self.add, T(label, "", "bold green"))
        out, last, end = "", None, None
        for last in stream_generate(self.model, self.tok, prompt, max_tokens=self.max_tokens, sampler=sampler):
            out += last.text
            if self.template and ("</answer>" in out or out.rstrip().endswith("User:")):
                if "</answer>" in out:
                    out, end = out.split("</answer>")[0] + "</answer>", "[finish: </answer>]"
                else:
                    out, end = out.rstrip()[:-5], "[finish: model started a new 'User:' turn - cut]"
                break
            self.app.call_from_thread(w.update, T(label, out, "bold green"))
            self.app.call_from_thread(self.log_w.scroll_end, animate=False)
            self.app.call_from_thread(self.update_status,
                                      f"| {last.generation_tokens} tok | {last.generation_tps:.0f} tok/s | mem {last.peak_memory:.1f}GB")
        if last is not None and end is None:
            end = f"[finish: {last.finish_reason}]"
            if last.finish_reason == "stop":
                end = f"[finish: stop — {self.tok.eos_token!r}]"
        self.app.call_from_thread(w.update, T(label, out, "bold green") + Text("\n" + (end or ""), style="dim"))
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": out}]
        self.busy = False


class ChatApp(App):
    TITLE = "mlx chat"

    def __init__(self, direct=None):
        super().__init__()
        self.direct = direct

    def on_mount(self):
        self.push_screen(PickerScreen())
        if self.direct:
            name, fmt = self.direct
            info = model_info(MODELS / name)
            fmt = fmt or info["fmt"]
            self.push_screen(ChatScreen(name, fmt, "" if fmt == "r1zero" else SYSTEM, 0.7, 1024))


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    direct = None
    if pos and (MODELS / pos[0]).is_dir():
        direct = (pos[0], "r1zero" if "--r1zero" in sys.argv else None)
    ChatApp(direct).run()
