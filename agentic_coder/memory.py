from __future__ import annotations
import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

"""
Memory Management: Session persistence and context retention
Similar to Claude Code's memory capabilities for maintaining context across sessions.
"""

@dataclass
class MemoryEntry:
    """A single memory entry."""
    timestamp: float
    type: str  # 'goal', 'action', 'result', 'conversation'
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class SessionMemory:
    """Memory for a single session."""
    session_id: str
    repo_path: str
    model_preset: str
    started_at: float
    last_active: float
    entries: List[MemoryEntry]
    goal: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class MemoryManager:
    """Manage session memory and persistence."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.memory_dir = os.path.join(repo_path, ".agentic_memory")
        self.current_session: Optional[SessionMemory] = None
        self.max_entries_per_session = 1000
        self.memory_retention_days = 30
        
        # Ensure memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)
        self._cleanup_old_sessions()
    
    def start_session(self, model_preset: str, session_id: Optional[str] = None) -> str:
        """Start a new session or resume existing one."""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        # Try to load existing session
        if os.path.exists(session_file):
            try:
                self.current_session = self._load_session(session_file)
                self.current_session.last_active = time.time()
                self._save_current_session()
                return session_id
            except Exception as e:
                print(f"Warning: Failed to load session {session_id}: {e}")
        
        # Create new session
        self.current_session = SessionMemory(
            session_id=session_id,
            repo_path=self.repo_path,
            model_preset=model_preset,
            started_at=time.time(),
            last_active=time.time(),
            entries=[],
            goal=None,
            context={}
        )
        
        self._save_current_session()
        return session_id
    
    def add_memory(self, type: str, content: str, metadata: Dict[str, Any] = None):
        """Add a memory entry to the current session."""
        if not self.current_session:
            return
        
        entry = MemoryEntry(
            timestamp=time.time(),
            type=type,
            content=content,
            metadata=metadata or {}
        )
        
        self.current_session.entries.append(entry)
        self.current_session.last_active = time.time()
        
        # Trim entries if too many
        if len(self.current_session.entries) > self.max_entries_per_session:
            # Keep recent entries and important ones
            important_types = {'goal', 'error', 'success'}
            recent_entries = self.current_session.entries[-500:]  # Keep last 500
            important_entries = [
                e for e in self.current_session.entries[:-500]
                if e.type in important_types
            ][-100:]  # Keep last 100 important
            
            self.current_session.entries = important_entries + recent_entries
        
        self._save_current_session()
    
    def set_goal(self, goal: str):
        """Set the current session goal."""
        if not self.current_session:
            return
        
        self.current_session.goal = goal
        self.add_memory('goal', goal, {'action': 'set_goal'})
    
    def get_goal(self) -> Optional[str]:
        """Get the current session goal."""
        return self.current_session.goal if self.current_session else None
    
    def update_context(self, key: str, value: Any):
        """Update session context."""
        if not self.current_session:
            return
        
        self.current_session.context[key] = value
        self._save_current_session()
    
    def get_context(self, key: str, default=None):
        """Get session context value."""
        if not self.current_session:
            return default
        return self.current_session.context.get(key, default)
    
    def get_recent_memories(self, limit: int = 50, types: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Get recent memory entries."""
        if not self.current_session:
            return []
        
        entries = self.current_session.entries
        
        if types:
            entries = [e for e in entries if e.type in types]
        
        return entries[-limit:]
    
    def search_memories(self, query: str, limit: int = 20) -> List[MemoryEntry]:
        """Search memory entries by content."""
        if not self.current_session:
            return []
        
        query_lower = query.lower()
        matches = []
        
        for entry in reversed(self.current_session.entries):
            if query_lower in entry.content.lower():
                matches.append(entry)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        if not self.current_session:
            return {}
        
        session = self.current_session
        entry_types = {}
        for entry in session.entries:
            entry_types[entry.type] = entry_types.get(entry.type, 0) + 1
        
        duration = session.last_active - session.started_at
        
        return {
            'session_id': session.session_id,
            'repo_path': session.repo_path,
            'model_preset': session.model_preset,
            'started_at': datetime.fromtimestamp(session.started_at).isoformat(),
            'duration_minutes': int(duration / 60),
            'total_entries': len(session.entries),
            'entry_types': entry_types,
            'current_goal': session.goal,
            'context_keys': list(session.context.keys())
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []
        
        if not os.path.exists(self.memory_dir):
            return sessions
        
        for filename in os.listdir(self.memory_dir):
            if filename.endswith('.json'):
                session_file = os.path.join(self.memory_dir, filename)
                try:
                    session = self._load_session(session_file)
                    sessions.append({
                        'session_id': session.session_id,
                        'started_at': datetime.fromtimestamp(session.started_at).isoformat(),
                        'last_active': datetime.fromtimestamp(session.last_active).isoformat(),
                        'model_preset': session.model_preset,
                        'entries_count': len(session.entries),
                        'current_goal': session.goal[:100] + '...' if session.goal and len(session.goal) > 100 else session.goal
                    })
                except Exception:
                    continue
        
        # Sort by last active (most recent first)
        sessions.sort(key=lambda x: x['last_active'], reverse=True)
        return sessions
    
    def load_session(self, session_id: str) -> bool:
        """Load a specific session by ID."""
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return False
        
        try:
            self.current_session = self._load_session(session_file)
            self.current_session.last_active = time.time()
            self._save_current_session()
            return True
        except Exception:
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_file = os.path.join(self.memory_dir, f"{session_id}.json")
        
        try:
            if os.path.exists(session_file):
                os.remove(session_file)
                if self.current_session and self.current_session.session_id == session_id:
                    self.current_session = None
                return True
        except Exception:
            pass
        
        return False
    
    def _load_session(self, session_file: str) -> SessionMemory:
        """Load session from file."""
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        entries = [MemoryEntry(**entry_data) for entry_data in data['entries']]
        data['entries'] = entries
        
        return SessionMemory(**data)
    
    def _save_current_session(self):
        """Save current session to file."""
        if not self.current_session:
            return
        
        session_file = os.path.join(self.memory_dir, f"{self.current_session.session_id}.json")
        
        # Convert to dict for JSON serialization
        data = asdict(self.current_session)
        
        try:
            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save session: {e}")
    
    def _cleanup_old_sessions(self):
        """Clean up old session files."""
        if not os.path.exists(self.memory_dir):
            return
        
        cutoff_time = time.time() - (self.memory_retention_days * 24 * 3600)
        
        for filename in os.listdir(self.memory_dir):
            if filename.endswith('.json'):
                session_file = os.path.join(self.memory_dir, filename)
                try:
                    # Check file modification time
                    if os.path.getmtime(session_file) < cutoff_time:
                        os.remove(session_file)
                except Exception:
                    continue

def format_memory_for_context(memory_manager: MemoryManager, limit: int = 20) -> str:
    """Format recent memories for inclusion in context."""
    if not memory_manager.current_session:
        return ""
    
    recent_memories = memory_manager.get_recent_memories(limit)
    if not recent_memories:
        return ""
    
    lines = ["RECENT CONTEXT:"]
    for entry in recent_memories:
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M")
        content = entry.content[:150] + "..." if len(entry.content) > 150 else entry.content
        lines.append(f"{timestamp} [{entry.type}] {content}")
    
    return "\n".join(lines)