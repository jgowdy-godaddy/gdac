"""Session management for Agentic Coder REPL with resumption functionality."""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class SessionMessage:
    """Represents a single message in a session."""
    role: str  # "user", "assistant", "tool"
    content: str
    timestamp: float
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None

@dataclass
class SessionMetadata:
    """Metadata for a session."""
    session_id: str
    created_at: float
    last_updated: float
    repo_path: str
    model: str
    title: str  # First user message or auto-generated
    message_count: int
    goal: Optional[str] = None

class SessionManager:
    """Manages saving and loading of conversation sessions."""
    
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.sessions_dir = Path.home() / ".agentic_coder" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / f"{session_id}.jsonl"
    
    def _get_metadata_file(self, session_id: str) -> Path:
        """Get the metadata file path for a session."""
        return self.sessions_dir / f"{session_id}.meta.json"
    
    def create_session(self, model: str, goal: str = None) -> str:
        """Create a new session and return the session ID."""
        timestamp = time.time()
        session_id = f"{int(timestamp)}_{hash(self.repo_path) % 10000:04d}"
        
        # Create title from goal or use default
        title = goal[:50] + "..." if goal and len(goal) > 50 else goal or "New Session"
        
        metadata = SessionMetadata(
            session_id=session_id,
            created_at=timestamp,
            last_updated=timestamp,
            repo_path=self.repo_path,
            model=model,
            title=title,
            message_count=0,
            goal=goal
        )
        
        self._save_metadata(metadata)
        return session_id
    
    def save_message(self, session_id: str, role: str, content: str, 
                     tool_call: Dict[str, Any] = None, tool_result: Dict[str, Any] = None):
        """Save a message to a session."""
        message = SessionMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            tool_call=tool_call,
            tool_result=tool_result
        )
        
        # Append to JSONL file
        session_file = self._get_session_file(session_id)
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")
        
        # Update metadata
        metadata = self._load_metadata(session_id)
        if metadata:
            metadata.last_updated = time.time()
            metadata.message_count += 1
            self._save_metadata(metadata)
    
    def load_session(self, session_id: str) -> List[SessionMessage]:
        """Load all messages from a session."""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return []
        
        messages = []
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    messages.append(SessionMessage(**data))
        
        return messages
    
    def _save_metadata(self, metadata: SessionMetadata):
        """Save session metadata."""
        meta_file = self._get_metadata_file(metadata.session_id)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
    
    def _load_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Load session metadata."""
        meta_file = self._get_metadata_file(session_id)
        if not meta_file.exists():
            return None
        
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return SessionMetadata(**data)
        except Exception:
            return None
    
    def list_sessions(self, limit: int = 20) -> List[SessionMetadata]:
        """List recent sessions for this repository."""
        sessions = []
        
        for meta_file in self.sessions_dir.glob("*.meta.json"):
            session_id = meta_file.name.replace(".meta.json", "")
            metadata = self._load_metadata(session_id)
            if metadata and metadata.repo_path == self.repo_path:
                sessions.append(metadata)
        
        # Sort by last updated, most recent first
        sessions.sort(key=lambda x: x.last_updated, reverse=True)
        return sessions[:limit]
    
    def get_latest_session(self) -> Optional[SessionMetadata]:
        """Get the most recent session for this repository."""
        sessions = self.list_sessions(limit=1)
        return sessions[0] if sessions else None
    
    def delete_session(self, session_id: str):
        """Delete a session and its metadata."""
        session_file = self._get_session_file(session_id)
        meta_file = self._get_metadata_file(session_id)
        
        if session_file.exists():
            session_file.unlink()
        if meta_file.exists():
            meta_file.unlink()
    
    def get_session_summary(self, session_id: str) -> str:
        """Get a brief summary of a session for display."""
        metadata = self._load_metadata(session_id)
        if not metadata:
            return "Unknown session"
        
        # Load first few messages to create summary
        messages = self.load_session(session_id)
        
        # Find first user message for context
        user_messages = [m for m in messages[:5] if m.role == "user"]
        preview = user_messages[0].content[:100] + "..." if user_messages else "No messages"
        
        created = datetime.fromtimestamp(metadata.created_at).strftime("%Y-%m-%d %H:%M")
        updated = datetime.fromtimestamp(metadata.last_updated).strftime("%Y-%m-%d %H:%M")
        
        return f"{metadata.title}\n  {preview}\n  {metadata.message_count} messages, created {created}, updated {updated}"
    
    def rebuild_conversation_history(self, session_id: str) -> str:
        """Rebuild the conversation history string from saved messages."""
        messages = self.load_session(session_id)
        
        history_parts = []
        for msg in messages:
            if msg.role == "user":
                history_parts.append(msg.content)
            elif msg.role == "assistant":
                history_parts.append(msg.content)
            elif msg.role == "tool" and msg.tool_result:
                history_parts.append(f"OBSERVATION: {msg.tool_result}")
        
        return "\n".join(history_parts)