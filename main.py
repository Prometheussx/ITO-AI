from __future__ import annotations

import base64
import asyncio
from typing import Any, cast, List, Dict
from typing_extensions import override

from textual import events
from audio_util import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, RichLog, Select, Label
from textual.reactive import reactive
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.binding import Binding

from openai import AsyncOpenAI
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
import os
import sounddevice as sd  # type: ignore


class DeviceSelectionScreen(Screen[int]):
    """A screen for selecting audio input device."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Confirm"),
    ]
    
    def __init__(self, devices: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.devices = devices
        
    @override
    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Select Audio Input Device:", id="device-label")
            options = [(f"{i}: {dev['name']}", i) for i, dev in enumerate(self.devices) if dev['max_input_channels'] > 0]
            yield Select(options, id="device-select", prompt="Choose a device...")
            
            with Horizontal(id="button-row"):
                yield Button("Cancel", id="cancel-button")
                yield Button("Confirm", id="confirm-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "confirm-button":
            selection = self.query_one("#device-select", Select).value
            if selection is not None:
                self.dismiss(selection)
    
    def on_select_changed(self, event: Select.Changed) -> None:
        self.query_one("#confirm-button", Button).disabled = event.value is None
        
    def action_cancel(self) -> None:
        self.dismiss(None)
        
    def action_confirm(self) -> None:
        selection = self.query_one("#device-select", Select).value
        if selection is not None:
            self.dismiss(selection)


class VoiceSelectionScreen(Screen[str]):
    """A screen for selecting AI voice."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Confirm"),
    ]
    
    def __init__(self) -> None:
        super().__init__()
        self.voices = [
            ("alloy", "Alloy - Balanced and neutral"),
            ("echo", "Echo - Soft and conversational"),
            ("fable", "Fable - British accent, warm"),
            ("onyx", "Onyx - Deep and authoritative"),
            ("shimmer", "Shimmer - Clear with higher pitch")
        ]
        
    @override
    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Select AI Voice:", id="voice-label")
            options = [(description, voice_id) for voice_id, description in self.voices]
            yield Select(options, id="voice-select", prompt="Choose a voice...")
            
            with Horizontal(id="button-row"):
                yield Button("Cancel", id="cancel-button")
                yield Button("Confirm", id="confirm-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "confirm-button":
            selection = self.query_one("#voice-select", Select).value
            if selection is not None:
                self.dismiss(selection)
    
    def on_select_changed(self, event: Select.Changed) -> None:
        self.query_one("#confirm-button", Button).disabled = event.value is None
        
    def action_cancel(self) -> None:
        self.dismiss(None)
        
    def action_confirm(self) -> None:
        selection = self.query_one("#voice-select", Select).value
        if selection is not None:
            self.dismiss(selection)


class SessionDisplay(Static):
    """A widget that shows the current session ID."""

    session_id = reactive("")

    @override
    def render(self) -> str:
        return f"Session ID: {self.session_id}" if self.session_id else "Connecting..."


class AudioStatusIndicator(Static):
    """A widget that shows the current audio recording status."""

    is_recording = reactive(False)
    device_name = reactive("Default")
    voice_name = reactive("Alloy")

    @override
    def render(self) -> str:
        status = (
            f"🔴 Recording with {self.device_name}... (Press K to stop)" 
            if self.is_recording 
            else f"⚪ Voice: {self.voice_name} | Press K to record, D for device, V for voice, Q to quit"
        )
        return status


class RealtimeApp(App[None]):
    CSS = """
        Screen {
            background: #1a1b26;  /* Dark blue-grey background */
        }

        Container {
            border: double rgb(91, 164, 91);
        }

        Horizontal {
            width: 100%;
            height: auto;
        }

        #input-container {
            height: 5;  /* Explicit height for input container */
            margin: 1 1;
            padding: 1 2;
        }

        Input {
            width: 80%;
            height: 3;  /* Explicit height for input */
        }

        Button {
            width: 20%;
            height: 3;  /* Explicit height for button */
            margin: 1 2;
        }

        #bottom-pane {
            width: 100%;
            height: 79%;  /* Reduced to make room for session display */
            border: round rgb(205, 133, 63);
            content-align: center middle;
        }

        #status-indicator {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        #session-display {
            height: 3;
            content-align: center middle;
            background: #2a2b36;
            border: solid rgb(91, 164, 91);
            margin: 1 1;
        }

        Static {
            color: white;
        }
        
        DeviceSelectionScreen, VoiceSelectionScreen {
            align: center middle;
        }
        
        DeviceSelectionScreen Container, VoiceSelectionScreen Container {
            width: 60;
            height: 15;
            background: #2a2b36;
            padding: 1;
        }
        
        #device-label, #voice-label {
            text-align: center;
            padding: 1;
        }
        
        #device-select, #voice-select {
            margin: 1 0;
            width: 100%;
        }
        
        #button-row {
            margin-top: 1;
            height: 3;
            align: center bottom;
            content-align: center middle;
        }
    """

    client: AsyncOpenAI
    should_send_audio: asyncio.Event
    audio_player: AudioPlayerAsync
    last_audio_item_id: str | None
    connection: AsyncRealtimeConnection | None
    session: Session | None
    connected: asyncio.Event
    device_id: int | None
    voice: str

    def __init__(self) -> None:
        super().__init__()
        self.connection = None
        self.session = None
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.audio_player = AudioPlayerAsync()
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.device_id = None
        self.audio_devices = self.get_audio_devices()
        

    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get a list of available audio devices."""
        return sd.query_devices()
    
    async def show_device_selection(self) -> None:
        """Show the device selection screen."""
        # Pause audio recording if it's happening
        was_recording = self.should_send_audio.is_set()
        if was_recording:
            self.should_send_audio.clear()
            status_indicator = self.query_one(AudioStatusIndicator)
            status_indicator.is_recording = False
            
        # Show the device selection screen
        result = await self.push_screen(DeviceSelectionScreen(self.audio_devices))
        
        if result is not None:
            self.device_id = result
            status_indicator = self.query_one(AudioStatusIndicator)
            device_name = self.audio_devices[result]["name"]
            status_indicator.device_name = f"{device_name[:20]}"

    async def show_voice_selection(self) -> None:
        """Show the voice selection screen."""
        # Pause audio recording if it's happening
        was_recording = self.should_send_audio.is_set()
        if was_recording:
            self.should_send_audio.clear()
            status_indicator = self.query_one(AudioStatusIndicator)
            status_indicator.is_recording = False
            
        # Show the voice selection screen
        result = await self.push_screen(VoiceSelectionScreen())
        
        if result is not None:
            self.voice = result
            if self.connection:
                try:
                    # Update the session with the new voice
                    await self.connection.session.update(session={
                        "voice": self.voice
                    })
                    status_indicator = self.query_one(AudioStatusIndicator)
                    # Find the voice description for display
                    for voice_id, description in VoiceSelectionScreen().voices:
                        if voice_id == self.voice:
                            status_indicator.voice_name = description.split(' - ')[0]
                            break
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.write(f"[green]Voice changed to {self.voice}[/green]")
                except Exception as e:
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.write(f"[red]Error changing voice: {str(e)}[/red]")

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Container():
            yield SessionDisplay(id="session-display")
            yield AudioStatusIndicator(id="status-indicator")
            yield RichLog(id="bottom-pane", wrap=True, highlight=True, markup=True)

    async def on_mount(self) -> None:
        self.run_worker(self.handle_realtime_connection())
        self.run_worker(self.send_mic_audio())

    async def handle_realtime_connection(self) -> None:
        async with self.client.beta.realtime.connect(model="gpt-4o-realtime-preview-2024-12-17") as conn:
            self.connection = conn
            self.connected.set()

            # Set initial session parameters including voice
            await conn.session.update(session={
                "turn_detection": {"type": "server_vad"},
                "voice": "echo"  # Set initial voice
            })

            acc_items: dict[str, Any] = {}

            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    session_display = self.query_one(SessionDisplay)
                    assert event.session.id is not None
                    session_display.session_id = event.session.id
                    continue

                if event.type == "session.updated":
                    self.session = event.session
                    continue

                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.audio_transcript.delta":
                    try:
                        text = acc_items[event.item_id]
                    except KeyError:
                        acc_items[event.item_id] = event.delta
                    else:
                        acc_items[event.item_id] = text + event.delta

                    # Clear and update the entire content because RichLog otherwise treats each delta as a new line
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.clear()
                    bottom_pane.write(acc_items[event.item_id])
                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        sent_audio = False

        status_indicator = self.query_one(AudioStatusIndicator)

        try:
            while True:
                await asyncio.sleep(0.1)  # Small delay to prevent CPU hogging
                
                if not self.should_send_audio.is_set():
                    continue
                
                # Create a new stream each time with possibly updated device info
                try:
                    read_size = int(SAMPLE_RATE * 0.02)
                    device_args = {"device": self.device_id} if self.device_id is not None else {}
                    
                    stream = sd.InputStream(
                        channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        dtype="int16",
                        **device_args
                    )
                    stream.start()
                    
                    status_indicator.is_recording = True
                    
                    # Send audio as long as recording is active
                    while self.should_send_audio.is_set():
                        if stream.read_available < read_size:
                            await asyncio.sleep(0.01)
                            continue

                        data, _ = stream.read(read_size)

                        connection = await self._get_connection()
                        if not sent_audio:
                            asyncio.create_task(connection.send({"type": "response.cancel"}))
                            sent_audio = True

                        await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))
                    
                    # Reset the recording state
                    status_indicator.is_recording = False
                    sent_audio = False
                    
                    # Stop and close the stream when no longer recording
                    stream.stop()
                    stream.close()
                    
                    # If we're in manual turn detection, commit the audio buffer
                    if self.session and self.session.turn_detection is None:
                        conn = await self._get_connection()
                        await conn.input_audio_buffer.commit()
                        await conn.response.create()
                    
                except Exception as e:
                    bottom_pane = self.query_one("#bottom-pane", RichLog)
                    bottom_pane.write(f"[red]Error with audio device: {str(e)}[/red]")
                    self.should_send_audio.clear()
                    status_indicator.is_recording = False
                    await asyncio.sleep(2)  # Give user time to read error
                
        except KeyboardInterrupt:
            pass

    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if event.key == "enter":
            if hasattr(self, "current_screen") and isinstance(self.screen, (DeviceSelectionScreen, VoiceSelectionScreen)):
                self.action_confirm()
            return

        if event.key == "q":
            self.exit()
            return

        if event.key == "k":
            status_indicator = self.query_one(AudioStatusIndicator)
            if status_indicator.is_recording:
                self.should_send_audio.clear()
            else:
                self.should_send_audio.set()
            return
            
        if event.key == "d":
            # Only allow device selection when not recording
            if not self.should_send_audio.is_set():
                await self.show_device_selection()
            return
            
        if event.key == "v":
            # Only allow voice selection when not recording
            if not self.should_send_audio.is_set():
                await self.show_voice_selection()
            return


if __name__ == "__main__":
    app = RealtimeApp()
    app.run()
