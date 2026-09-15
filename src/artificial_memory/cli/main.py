from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Optional PostgreSQL support
try:
    from artificial_memory.storage import PostgresMemoryStore
    _HAS_POSTGRES = True
except ImportError:
    PostgresMemoryStore = None  # type: ignore
    _HAS_POSTGRES = False
from artificial_memory.context.ir_compiler import IRCompressor
from artificial_memory.memory.compiler import CompilerPipelineWrapper
from artificial_memory.memory.file_writer import ConversationFileWriter, MemoryFileWriter

# Optional pgvector support
try:
    from artificial_memory.memory.pgvector_search import (
        PgVectorSearchEngine,
        create_pgvector_search_engine,
    )
    _HAS_PGVECTOR = True
except ImportError:
    PgVectorSearchEngine = None  # type: ignore
    create_pgvector_search_engine = None  # type: ignore
    _HAS_PGVECTOR = False
from artificial_memory.core.models import (
    ConversationStatus,
    MemoryStatus,
    MemoryType,
    RecallLevel,
    ResolutionLevel,
)
from artificial_memory.memory.style import ConversationStyle
from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

console = Console()


class ArtificialMemoryCLI:
    """Main CLI application."""

    def __init__(self, db_path: Path | None, memory_files_path: Path, use_postgres: bool = False):
        self.db_path = db_path
        self.memory_files_path = memory_files_path
        self.use_postgres = use_postgres

        # Unified Runtime Facade
        config = RuntimeConfig(
            database_path=str(db_path) if db_path else "memory.db",
            use_postgres=use_postgres,
            database_url=os.environ.get("DATABASE_URL") if use_postgres else None,
            memory_files_path=str(memory_files_path),
        )
        self.runtime = ArtificialMemoryRuntime(config)

        # Delegate sub-components to the runtime facade
        self.store = self.runtime.store
        self.conversation_manager = self.runtime.conversation_manager
        self.topic_classifier = self.runtime.topic_classifier
        self.compressor = self.runtime.compressor
        self.recall_engine = self.runtime.recall_engine
        self.context_builder = self.runtime.context_builder
        self.memory_compiler = self.runtime.memory_compiler
        self.pipeline_wrapper = CompilerPipelineWrapper(
            self.store, self.compressor, self.topic_classifier
        )
        self.association_engine = self.runtime.association_engine
        self.temporal_engine = self.runtime.temporal_engine
        self.confidence_engine = self.runtime.confidence_engine
        self.style_engine = self.runtime.style_engine
        self.human_recall_engine = self.runtime.human_recall_engine
        self.vector_search_engine = self.runtime.vector_search_engine
        self.ir_compiler = self.runtime.ir_compiler
        self.ir_compressor = IRCompressor(max_tokens=4000)
        self.consolidation_engine = self.runtime.consolidation_engine
        self.consolidation_scheduler = self.runtime.consolidation_scheduler
        self.memory_writer = MemoryFileWriter(self.store, memory_files_path / "topics")
        self.conversation_writer = ConversationFileWriter(self.store, memory_files_path / "conversations")

        self._current_topic: str | None = None
        self._load_active_conversation()

    def _load_active_conversation(self):
        """Load active conversation from database."""
        # Find the most recent active conversation
        conversations = self.store.list_conversations(status=ConversationStatus.ACTIVE)
        if conversations:
            self.conversation_manager.logger.load_conversation(conversations[0])

    def _save_active_conversation(self):
        """Save active conversation state."""
        if self.conversation_manager.current_conversation:
            conv = self.conversation_manager.current_conversation
            conv.message_count = self.conversation_manager.logger._message_sequence
            self.store.update_conversation(conv)

    def close(self):
        self._save_active_conversation()
        self.runtime.close()


@click.group()
@click.option('--db', default='memory.db', help='Database file path (SQLite)')
@click.option('--memory-dir', default='memory_files', help='Memory files directory')
@click.option('--postgres/--sqlite', default=False, help='Use PostgreSQL instead of SQLite', envvar='USE_POSTGRES')
@click.pass_context
def cli(ctx, db, memory_dir, postgres):
    """Artificial Memory / Context Runtime - Cognitive Runtime for LLMs"""
    db_path = Path(db) if not postgres else None
    memory_files_path = Path(memory_dir)

    ctx.obj = ArtificialMemoryCLI(db_path, memory_files_path, use_postgres=postgres)
    ctx.call_on_close(lambda: ctx.obj.close())


@cli.command()
@click.argument('topic_path')
@click.option('--project', default='default', help='Project name')
@click.option('--title', help='Conversation title')
@click.pass_obj
def start(app: ArtificialMemoryCLI, topic_path: str, project: str, title: str | None):
    """Start a new conversation."""
    conversation = app.conversation_manager.start_conversation(topic_path, project, title)
    app._current_topic = topic_path
    console.print(f"[green]Started conversation:[/green] {conversation.id}")
    console.print(f"Topic: {topic_path} (Project: {project})")
    if title:
        console.print(f"Title: {title}")


@cli.command()
@click.argument('message')
@click.pass_obj
def user(app: ArtificialMemoryCLI, message: str):
    """Log a user message."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation. Use 'start' first.[/red]")
        return

    msg = app.conversation_manager.log_user(message)
    console.print(f"[blue]User:[/blue] {message[:100]}...")

    # Process with memory compiler
    current_memories = app.store.get_memories(
        topic_id=app.conversation_manager.current_conversation.topic_id,
        is_current=True,
        status=MemoryStatus.ACTIVE
    )
    new_memories = app.memory_compiler.process_message(msg, app.conversation_manager.current_conversation, current_memories)
    if new_memories:
        console.print(f"[dim]Created {len(new_memories)} new memories[/dim]")


@cli.command()
@click.argument('message')
@click.pass_obj
def assistant(app: ArtificialMemoryCLI, message: str):
    """Log an assistant message."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation. Use 'start' first.[/red]")
        return

    msg = app.conversation_manager.log_assistant(message)
    console.print(f"[green]Assistant:[/green] {message[:100]}...")

    # Process with memory compiler
    current_memories = app.store.get_memories(
        topic_id=app.conversation_manager.current_conversation.topic_id,
        is_current=True,
        status=MemoryStatus.ACTIVE
    )
    new_memories = app.memory_compiler.process_message(msg, app.conversation_manager.current_conversation, current_memories)
    if new_memories:
        console.print(f"[dim]Created {len(new_memories)} new memories[/dim]")


@cli.command()
@click.pass_obj
def end(app: ArtificialMemoryCLI):
    """End current conversation and compile memories."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation.[/red]")
        return

    conversation = app.conversation_manager.end_conversation()
    console.print(f"[green]Ended conversation:[/green] {conversation.id}")

    # Compile full conversation
    new_memories = app.memory_compiler.compile_conversation(conversation)
    console.print(f"[dim]Compiled {len(new_memories)} memories from conversation[/dim]")

    # Write memory files
    topic = app.store.get_topic(conversation.topic_id)
    if topic:
        files = app.memory_writer.write_all(topic)
        console.print(f"[dim]Wrote {len(files)} memory files[/dim]")

        # Write conversation file
        conv_file = app.conversation_writer.write_conversation(conversation.id)
        console.print(f"[dim]Wrote conversation file: {conv_file.name}[/dim]")

    # Run consolidation cycle
    stats = app.consolidation_scheduler.force_run()
    console.print(f"[dim]Consolidation: {stats}[/dim]")


@cli.command()
@click.option('--topic', help='Topic path to update')
@click.pass_obj
def update_hierarchy(app: ArtificialMemoryCLI, topic: str | None):
    """Update resolution hierarchy for topic (rebuild current/timeline/decisions)."""
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topics = [t]
    else:
        topics = app.store.list_topics()

    for t in topics:
        console.print(f"[bold]Updating hierarchy for {t.path}...[/bold]")

        # Recompile all conversations in topic
        conversations = app.store.list_conversations(topic_id=t.id)
        total_memories = 0
        for conv in conversations:
            if conv.status.value == "completed":
                memories = app.memory_compiler.compile_conversation(conv)
                total_memories += len(memories)

        # Write memory files
        files = app.memory_writer.write_all(t)

        # Run consolidation
        stats = app.consolidation_engine.consolidate_topic(t.id)

        console.print(f"  [green]Done:[/green] {len(conversations)} conversations, {total_memories} memories, {len(files)} files")
        console.print(f"  [dim]Consolidation: {stats}[/dim]")


@cli.command()
@click.pass_obj
def memory(app: ArtificialMemoryCLI):
    """Show current memory state (/memory command)."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation.[/red]")
        return

    topic_id = app.conversation_manager.current_conversation.topic_id
    memories = app.store.get_memories(topic_id=topic_id, is_current=True, status=MemoryStatus.ACTIVE)

    table = Table(title="Current Memories")
    table.add_column("Type", style="cyan")
    table.add_column("Resolution", style="magenta")
    table.add_column("Importance", style="yellow")
    table.add_column("Confidence", style="green")
    table.add_column("Preview", style="white")

    for mem in memories:
        preview = mem.content[:80].replace('\n', ' ') + "..." if len(mem.content) > 80 else mem.content
        table.add_row(
            mem.memory_type.value,
            mem.resolution.name,
            f"{mem.importance:.2f}",
            f"{mem.confidence:.2f}",
            preview
        )

    console.print(table)


@cli.command()
@click.pass_obj
def context(app: ArtificialMemoryCLI):
    """Show current context being sent to LLM (/context command)."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation.[/red]")
        return

    topic_id = app.conversation_manager.current_conversation.topic_id
    current_memories = app.store.get_memories(
        topic_id=topic_id,
        is_current=True,
        status=MemoryStatus.ACTIVE
    )

    # Build context
    query = "current context"
    context_str = app.context_builder.build_context(query, topic_id, max_tokens=4000, current_memories=current_memories)
    stats = app.context_builder.get_context_stats()

    console.print(Panel(
        f"[bold]Raw Tokens:[/bold] {stats.raw_tokens}\n"
        f"[bold]Effective Tokens:[/bold] {stats.effective_tokens}\n"
        f"[bold]Compression Ratio:[/bold] {stats.compression_ratio:.2f}x\n"
        f"[bold]Parts:[/bold] {stats.parts_count} \u2192 {stats.selected_parts}\n"
        f"[bold]Tier Distribution:[/bold] {stats.tier_distribution}\n"
        f"[bold]Truncated:[/bold] {stats.truncated_parts}",
        title="Context Stats"
    ))

    console.print(Panel(context_str[:3000] + ("..." if len(context_str) > 3000 else ""), title="Context Preview"))


@cli.command()
@click.argument('query')
@click.option('--level', type=click.Choice(['0', '1', '2', '3', '4']), default='0', help='Recall level')
@click.option('--max-tokens', default=4000, help='Max tokens to return')
@click.pass_obj
def recall(app: ArtificialMemoryCLI, query: str, level: str, max_tokens: int):
    """Recall memories (/recall command)."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation.[/red]")
        return

    topic_id = app.conversation_manager.current_conversation.topic_id
    recall_level = RecallLevel(int(level))

    memories, tokens = app.recall_engine.recall(query, topic_id, recall_level, max_tokens)

    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"[bold]Level:[/bold] {recall_level.name} ({recall_level.value})")
    console.print(f"[bold]Retrieved:[/bold] {len(memories)} memories, ~{tokens} tokens")
    console.print()

    for i, mem in enumerate(memories):
        console.print(Panel(
            mem.content,
            title=f"[{i+1}] {mem.memory_type.value} | {mem.resolution.name} | Imp: {mem.importance:.2f} | Conf: {mem.confidence:.2f}",
            border_style="blue"
        ))


@cli.command()
@click.argument('memory_id', type=int)
@click.option('--target', type=click.Choice(['0', '1', '2', '3', '4', '5']), help='Target resolution level')
@click.pass_obj
def expand(app: ArtificialMemoryCLI, memory_id: int, target: str | None):
    """Expand memory to higher resolution (/expand command)."""
    memory = app.store.get_memory(memory_id)
    if not memory:
        console.print(f"[red]Memory {memory_id} not found[/red]")
        return

    if target is None:
        # Show available versions
        versions = app.store.get_memory_versions(memory_id)
        console.print(f"Available resolutions for memory {memory_id}:")
        for v in versions:
            console.print(f"  {v.resolution.name} (ratio: {v.compression_ratio:.2f}x)")
        return

    target_res = ResolutionLevel(int(target))
    expanded = app.recall_engine.expand_resolution(memory, target_res)

    if expanded:
        console.print(Panel(
            expanded.content,
            title=f"Expanded to {target_res.name}",
            border_style="green"
        ))
    else:
        console.print(f"[yellow]No version available at {target_res.name}[/yellow]")


@cli.command()
@click.pass_obj
def timeline(app: ArtificialMemoryCLI):
    """Show topic timeline (/timeline command)."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation.[/red]")
        return

    topic_id = app.conversation_manager.current_conversation.topic_id
    memories = app.store.get_memories(
        topic_id=topic_id,
        memory_type=MemoryType.TIMELINE,
        status=MemoryStatus.ACTIVE
    )
    memories.sort(key=lambda m: m.valid_from or m.created_at)

    for mem in memories:
        date_str = (mem.valid_from or mem.created_at).strftime('%Y-%m-%d')
        console.print(f"[bold]{date_str}[/bold]: {mem.content[:200]}")


@cli.command()
@click.argument('memory_id', type=int)
@click.option('--full', is_flag=True, help='Show full provenance including source conversation')
@click.pass_obj
def trace(app: ArtificialMemoryCLI, memory_id: int, full: bool):
    """Trace memory provenance (/trace command)."""
    memory = app.store.get_memory(memory_id)
    if not memory:
        console.print(f"[red]Memory {memory_id} not found[/red]")
        return

    if full:
        provenance = app.recall_engine.get_full_provenance(memory)

        console.print(f"[bold]Full Provenance for Memory {memory_id}:[/bold]")
        console.print(f"  Type: {memory.memory_type.value}")
        console.print(f"  Current Resolution: {memory.resolution.name}")
        console.print(f"  Status: {memory.status.value}")
        console.print(f"  Importance: {memory.importance:.2f}")
        console.print(f"  Confidence: {memory.confidence:.2f}")
        console.print()

        # Show version chain
        console.print("[bold]Version Chain (Progressive Recall):[/bold]")
        for i, mem in enumerate(provenance["versions"]):
            marker = " → " if i > 0 else "   "
            console.print(f"  {marker}{mem.resolution.name}: {mem.content[:120]}...")
        console.print()

        # Show source conversation
        if provenance["source_conversation"]:
            conv = provenance["source_conversation"]
            console.print("[bold]Source Conversation:[/bold]")
            console.print(f"  ID: {conv.id}")
            console.print(f"  Title: {conv.title}")
            console.print(f"  Date: {conv.started_at.strftime('%Y-%m-%d %H:%M')}")
            console.print(f"  Messages: {conv.message_count}")
            console.print()

            if provenance["source_messages"]:
                console.print("[bold]Source Messages:[/bold]")
                for msg in provenance["source_messages"]:
                    role_label = "[User]" if msg.role.value == "user" else "[Asst]"
                    preview = msg.content[:100].replace('\n', ' ')
                    console.print(f"  {role_label} [{msg.sequence_num}] {preview}...")

                if provenance.get("exact_source_message"):
                    msg = provenance["exact_source_message"]
                    console.print("\n  [yellow]^^ Exact source message for this memory[/yellow]")
        else:
            console.print("[yellow]No source conversation linked[/yellow]")
    else:
        chain = app.recall_engine.get_memory_provenance(memory)

        console.print(f"[bold]Provenance for Memory {memory_id}:[/bold]")
        for i, mem in enumerate(chain):
            console.print(f"  {i}. {mem.resolution.name}: {mem.content[:100]}...")


@cli.command()
@click.option('--compile', is_flag=True, help='Compile current conversation to IR')
@click.option('--compress', is_flag=True, help='Show compressed IR')
@click.option('--decompile', is_flag=True, help='Decompile IR back to natural language')
@click.pass_obj
def dump_ir(app: ArtificialMemoryCLI, compile: bool, compress: bool, decompile: bool):
    """Dump Context IR (/dump_ir command)."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation.[/red]")
        return

    conv = app.conversation_manager.current_conversation

    if compile or not app.store.get_context_ir(conv.id):
        # Compile conversation to IR
        messages = app.store.get_messages(conv.id)
        if not messages:
            console.print("[yellow]No messages to compile[/yellow]")
            return

        console.print(f"[dim]Compiling {len(messages)} messages to IR...[/dim]")
        ir_sequence = app.ir_compiler.compile_to_ir(messages)
        ir_sequence.conversation_id = conv.id
        ir_sequence.topic_id = conv.topic_id

        # Store IR units in database
        for unit in ir_sequence.units:
            from artificial_memory.core.models import ContextIR
            ir = ContextIR(
                conversation_id=conv.id,
                ir_type=unit.ir_type,
                ir_key=unit.ir_key,
                ir_value=unit.ir_value,
                sequence_num=unit.sequence_num,
            )
            app.store.add_context_ir(ir)

        console.print(f"[green]Compiled {len(ir_sequence.units)} IR units[/green]")

        if compress:
            compressed = app.ir_compressor.compress(ir_sequence)
            console.print(f"[dim]Compressed: {ir_sequence.total_tokens()} → {compressed.total_tokens()} tokens ({len(ir_sequence.units)} → {len(compressed.units)} units)[/dim]")
            ir_sequence = compressed

        if decompile:
            reconstructed = app.ir_compiler.decompile_from_ir(ir_sequence)
            console.print(Panel(reconstructed[:3000] + ("..." if len(reconstructed) > 3000 else ""), title="Decompiled"))
            return

    # Display stored IR
    ir_units = app.store.get_context_ir(conv.id)

    if not ir_units:
        console.print("[yellow]No IR units found. Use --compile to generate.[/yellow]")
        return

    # Summary stats
    total_tokens = sum(len(u.ir_key) + len(u.ir_value or "") for u in ir_units) // 3
    by_type = {}
    for ir in ir_units:
        by_type[ir.ir_type] = by_type.get(ir.ir_type, 0) + 1

    console.print(Panel(
        f"[bold]Conversation:[/bold] {conv.id}\n"
        f"[bold]Total IR Units:[/bold] {len(ir_units)}\n"
        f"[bold]Estimated Tokens:[/bold] ~{total_tokens}\n"
        f"[bold]Types:[/bold] {', '.join(f'{k}({v})' for k,v in by_type.items())}",
        title="Context IR Summary"
    ))

    # Detailed table
    table = Table(title="IR Units")
    table.add_column("Seq", style="yellow", width=4)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Key", style="magenta", width=10)
    table.add_column("Value", style="white")

    for ir in ir_units:
        value = ir.ir_value or ""
        if len(value) > 80:
            value = value[:77] + "..."
        table.add_row(str(ir.sequence_num), ir.ir_type, ir.ir_key, value)

    console.print(table)

    # Show compact representation
    if ir_units:
        compact = ' '.join(f"{u.ir_key}:{u.ir_value}" if u.ir_value else u.ir_key for u in ir_units)
        console.print(Panel(compact[:500] + ("..." if len(compact) > 500 else ""), title="Compact Representation"))


@cli.command()
@click.pass_obj
def stats(app: ArtificialMemoryCLI):
    """Show system statistics."""
    if not app.conversation_manager.current_conversation:
        console.print("[red]No active conversation.[/red]")
        return

    topic_id = app.conversation_manager.current_conversation.topic_id

    # Memory stats
    all_memories = app.store.get_memories(topic_id=topic_id, limit=1000)

    table = Table(title="Memory Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Memories", str(len(all_memories)))

    by_type = {}
    by_resolution = {}
    by_status = {}

    for mem in all_memories:
        by_type[mem.memory_type.value] = by_type.get(mem.memory_type.value, 0) + 1
        by_resolution[mem.resolution.name] = by_resolution.get(mem.resolution.name, 0) + 1
        by_status[mem.status.value] = by_status.get(mem.status.value, 0) + 1

    for k, v in sorted(by_type.items()):
        table.add_row(f"  Type: {k}", str(v))
    for k, v in sorted(by_resolution.items()):
        table.add_row(f"  Resolution: {k}", str(v))
    for k, v in sorted(by_status.items()):
        table.add_row(f"  Status: {k}", str(v))

    console.print(table)

    # Token usage
    usage = app.store.get_token_usage()
    if usage:
        total_in = sum(u.input_tokens for u in usage)
        total_out = sum(u.output_tokens for u in usage)
        console.print(f"\n[bold]Token Usage:[/bold] In: {total_in}, Out: {total_out}, Total: {total_in + total_out}")


@cli.command()
@click.pass_obj
def compress(app: ArtificialMemoryCLI):
    """Force consolidation (/compress command)."""
    results = app.consolidation_scheduler.force_run()
    console.print(f"[green]Consolidation complete:[/green] {results}")


@cli.command()
@click.pass_obj
def consolidation_status(app: ArtificialMemoryCLI):
    """Show consolidation engine status and statistics."""
    stats = app.consolidation_engine.run_consolidation_cycle()  # Dry run to get stats
    console.print(Panel.fit(
        f"[bold]Topics Processed:[/bold] {stats['topics_processed']}\n"
        f"[bold]Memories Evaluated:[/bold] {stats['evaluated']}\n"
        f"[bold]Status Changed:[/bold] {stats['status_changed']}\n"
        f"[bold]Compressed:[/bold] {stats['compressed']}\n"
        f"[bold]Archived:[/bold] {stats['archived']}\n"
        f"[bold]Deep Archived:[/bold] {stats['deep_archived']}",
        title="Consolidation Status"
    ))


@cli.command()
@click.option('--topic', help='Topic path to consolidate')
@click.pass_obj
def consolidate_topic(app: ArtificialMemoryCLI, topic: str | None):
    """Run consolidation for specific topic."""
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        stats = app.consolidation_engine.consolidate_topic(t.id)
    else:
        stats = app.consolidation_scheduler.force_run()
    console.print(f"[green]Consolidation complete:[/green] {stats}")


@cli.command()
@click.option('--topic', help='Topic path to analyze')
@click.pass_obj
def associate(app: ArtificialMemoryCLI, topic: str | None):
    """Analyze and create semantic associations for topic."""
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topics = [t]
    else:
        topics = app.store.list_topics()

    total_created = 0
    for t in topics:
        console.print(f"[bold]Analyzing associations for {t.path}...[/bold]")
        associations = app.association_engine.analyze_and_create_associations(t.id)
        console.print(f"  [green]Created {len(associations)} associations[/green]")
        total_created += len(associations)

    console.print(f"[green]Total associations created: {total_created}[/green]")


@cli.command()
@click.argument('memory_id', type=int)
@click.option('--depth', default=2, help='Traversal depth')
@click.option('--min-strength', default=0.3, help='Minimum association strength')
@click.pass_obj
def associations(app: ArtificialMemoryCLI, memory_id: int, depth: int, min_strength: float):
    """Show associated memories for a memory."""
    memory = app.store.get_memory(memory_id)
    if not memory:
        console.print(f"[red]Memory {memory_id} not found[/red]")
        return

    console.print(f"[bold]Associations for Memory {memory_id} ({memory.memory_type.value}):[/bold]")

    # Get direct associations
    related = app.store.get_related_memories(memory_id, min_strength)

    if not related:
        console.print("[yellow]No associations found[/yellow]")
        return

    table = Table(title=f"Direct Associations (strength >= {min_strength})")
    table.add_column("Target ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Strength", style="yellow")
    table.add_column("Preview", style="white")

    for mem, assoc in related:
        preview = mem.content[:80].replace('\n', ' ') + "..."
        table.add_row(str(mem.id), assoc.association_type.value, f"{assoc.strength:.2f}", preview)

    console.print(table)

    # Show association path if requested
    if depth > 1:
        console.print(f"\n[bold]Exploring depth {depth}...[/bold]")
        visited = set([memory_id])
        current_level = [(memory_id, 0)]

        for d in range(1, depth + 1):
            next_level = []
            for mem_id, _ in current_level:
                related = app.store.get_related_memories(mem_id, min_strength)
                for mem, assoc in related:
                    if mem.id not in visited:
                        visited.add(mem.id)
                        next_level.append((mem.id, assoc.strength))
                        preview = mem.content[:60].replace('\n', ' ')
                        console.print(f"  {'  ' * d}└─ [{assoc.strength:.2f}] {mem.id}: {preview}...")

            current_level = next_level
            if not current_level:
                break


@cli.command()
@click.argument('source_id', type=int)
@click.argument('target_id', type=int)
@click.option('--max-depth', default=3, help='Maximum search depth')
@click.pass_obj
def assoc_path(app: ArtificialMemoryCLI, source_id: int, target_id: int, max_depth: int):
    """Find association path between two memories."""
    path = app.association_engine.find_association_path(source_id, target_id, max_depth)

    if path:
        console.print(f"[green]Found path (length {len(path)}):[/green]")
        for i, assoc in enumerate(path):
            direction = "→" if assoc.source_memory_id != target_id else "←"
            console.print(f"  {i+1}. {assoc.source_memory_id} {direction} {assoc.target_memory_id} [{assoc.association_type.value}, {assoc.strength:.2f}]")
    else:
        console.print(f"[yellow]No path found within depth {max_depth}[/yellow]")


@cli.command()
@click.option('--topic', help='Topic path')
@click.option('--timestamp', help='Timestamp (ISO format), defaults to now')
@click.option('--include-archived', is_flag=True, help='Include archived memories')
@click.pass_obj
def temporal_state(app: ArtificialMemoryCLI, topic: str | None, timestamp: str | None, include_archived: bool):
    """Show temporal state at a specific timestamp."""
    from datetime import datetime

    ts = datetime.fromisoformat(timestamp) if timestamp else datetime.now()

    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        state = app.temporal_engine.get_state_at(ts, t.id, include_archived)
    else:
        state = app.temporal_engine.get_state_at(ts, include_archived=include_archived)

    console.print(Panel.fit(
        f"[bold]Timestamp:[/bold] {ts.isoformat()}\n"
        f"[bold]Active Memories:[/bold] {len(state.active_memories)}\n"
        f"[bold]Active Decisions:[/bold] {len(state.active_decisions)}",
        title="Temporal State"
    ))

    if state.active_memories:
        table = Table(title="Active Memories")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Resolution", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Preview", style="white")

        for mem in state.active_memories[:20]:
            table.add_row(str(mem.id), mem.memory_type.value, mem.resolution.name, mem.status.value, mem.content[:80].replace('\n', ' ') + "...")

        console.print(table)

    if state.active_decisions:
        table = Table(title="Active Decisions")
        table.add_column("ID", style="cyan")
        table.add_column("Decision", style="white")
        table.add_column("Confidence", style="yellow")

        for dec in state.active_decisions[:10]:
            table.add_row(str(dec.id), dec.decision_text[:80], f"{dec.confidence:.2f}")

        console.print(table)


@cli.command()
@click.option('--topic', required=True, help='Topic path')
@click.option('--start', required=True, help='Start timestamp (ISO format)')
@click.option('--end', required=True, help='End timestamp (ISO format)')
@click.pass_obj
def temporal_changes(app: ArtificialMemoryCLI, topic: str, start: str, end: str):
    """Show changes between two timestamps."""
    from datetime import datetime

    t = app.conversation_manager.get_or_create_topic(topic)
    start_ts = datetime.fromisoformat(start)
    end_ts = datetime.fromisoformat(end)

    changes = app.temporal_engine.get_changes_between(start_ts, end_ts, t.id)

    console.print(Panel.fit(
        f"[bold]Period:[/bold] {start} → {end}\n"
        f"[bold]Added:[/bold] {len(changes['added'])}\n"
        f"[bold]Removed:[/bold] {len(changes['removed'])}\n"
        f"[bold]Modified:[/bold] {len(changes['modified'])}",
        title="Temporal Changes"
    ))

    if changes['added']:
        console.print("[bold green]Added:[/bold green]")
        for item in changes['added']:
            console.print(f"  + {item['type']} ({item['id']}): {item['preview']}")

    if changes['removed']:
        console.print("[bold red]Removed:[/bold red]")
        for item in changes['removed']:
            console.print(f"  - {item['type']} ({item['id']}): {item['preview']}")

    if changes['modified']:
        console.print("[bold yellow]Modified:[/bold yellow]")
        for item in changes['modified']:
            console.print(f"  ~ {item['memory_id']}: content/status changed")


@cli.command()
@click.option('--topic', required=True, help='Topic path')
@click.option('--start', help='Start timestamp (ISO format)')
@click.option('--end', help='End timestamp (ISO format)')
@click.pass_obj
def temporal_timeline(app: ArtificialMemoryCLI, topic: str, start: str | None, end: str | None):
    """Show timeline of changes for a topic."""
    from datetime import datetime

    t = app.conversation_manager.get_or_create_topic(topic)

    start_ts = datetime.fromisoformat(start) if start else None
    end_ts = datetime.fromisoformat(end) if end else None

    timeline = app.temporal_engine.get_timeline_for_topic(t.id, start_ts, end_ts)

    console.print(Panel.fit(
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Events:[/bold] {len(timeline)}",
        title="Temporal Timeline"
    ))

    table = Table(title="Timeline")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Memory ID", style="magenta")
    table.add_column("Type", style="yellow")
    table.add_column("Resolution", style="green")
    table.add_column("Status", style="blue")
    table.add_column("Preview", style="white")

    for event in timeline[:50]:
        table.add_row(
            event["timestamp"],
            str(event["memory_id"]),
            event["type"],
            event["resolution"],
            event["status"],
            event["preview"][:80]
        )

    console.print(table)


@cli.command()
@click.argument('query')
@click.option('--topic', help='Topic path')
@click.option('--level', type=click.Choice(['0', '1', '2', '3', '4']), default='2', help='Recall level')
@click.pass_obj
def confidence(app: ArtificialMemoryCLI, query: str, topic: str | None, level: str):
    """Compute and show confidence for a query."""
    from artificial_memory.core.models import RecallLevel

    recall_level = RecallLevel(int(level))

    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topic_id = t.id
    else:
        topic_id = None

    # Recall memories
    memories, tokens = app.recall_engine.recall(query, topic_id, recall_level, max_tokens=4000)

    if not memories:
        console.print("[yellow]No memories found for query[/yellow]")
        return

    # Compute confidence
    confidence = app.confidence_engine.compute_overall_confidence(query, memories, recall_level)

    console.print(Panel.fit(
        f"[bold]Query:[/bold] {query}\n"
        f"[bold]Recall Level:[/bold] {recall_level.name}\n"
        f"[bold]Memories Found:[/bold] {len(memories)}\n"
        f"[bold]Overall Confidence:[/bold] {confidence.overall:.2f} ({confidence.level.value})\n"
        f"[bold]Natural Language:[/bold] {confidence.to_natural_language()}",
        title="Confidence Assessment"
    ))

    # Show breakdown
    table = Table(title="Confidence Breakdown")
    table.add_column("Component", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Weight", style="magenta")

    table.add_row("Memory Confidence", f"{confidence.memory_confidence:.2f}", "40%")
    table.add_row("Retrieval Confidence", f"{confidence.retrieval_confidence:.2f}", "30%")
    table.add_row("Temporal Confidence", f"{confidence.temporal_confidence:.2f}", "15%")
    table.add_row("Source Confidence", f"{confidence.source_confidence:.2f}", "15%")

    console.print(table)

    # Show memories with individual confidence
    console.print("\n[bold]Memories with Individual Confidence:[/bold]")
    for mem in memories:
        mem_conf = app.confidence_engine.compute_memory_confidence(mem)
        console.print(f"  [{mem_conf:.2f}] {mem.id}: {mem.memory_type.value} - {mem.content[:80].replace(chr(10), ' ')}...")


@cli.command()
@click.argument('memory_id', type=int)
@click.pass_obj
def mem_confidence(app: ArtificialMemoryCLI, memory_id: int):
    """Show detailed confidence breakdown for a memory."""
    memory = app.store.get_memory(memory_id)
    if not memory:
        console.print(f"[red]Memory {memory_id} not found[/red]")
        return

    conf = app.confidence_engine.compute_memory_confidence(memory)

    console.print(Panel.fit(
        f"[bold]Memory ID:[/bold] {memory.id}\n"
        f"[bold]Type:[/bold] {memory.memory_type.value}\n"
        f"[bold]Resolution:[/bold] {memory.resolution.name}\n"
        f"[bold]Status:[/bold] {memory.status.value}\n"
        f"[bold]Base Confidence:[/bold] {memory.confidence:.2f}\n"
        f"[bold]Computed Confidence:[/bold] {conf:.2f}",
        title="Memory Confidence"
    ))

    # Show factors
    table = Table(title="Confidence Factors")
    table.add_column("Factor", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Base Confidence", f"{memory.confidence:.2f}")
    table.add_row("Status Modifier", f"{1.0 if memory.status.value == 'active' else 0.9 if memory.status.value == 'dormant' else 0.8 if memory.status.value == 'compressed' else 0.6 if memory.status.value == 'archived' else 0.4}")
    table.add_row("Resolution Modifier", f"{1.0 - memory.resolution.value * 0.05:.2f}")
    table.add_row("Age Modifier", f"{max(0.5, 1.0 - (datetime.now() - memory.created_at).days / 365 * 0.3):.2f}")
    table.add_row("Access Modifier", f"{min(1.0, 1.0 + memory.access_count * 0.02):.2f}")

    console.print(table)


@cli.command()
@click.option('--topic', help='Topic path')
@click.pass_obj
def style_profile(app: ArtificialMemoryCLI, topic: str | None):
    """Show conversation style profile for a topic."""
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topics = [t]
    else:
        topics = app.store.list_topics()

    for t in topics:
        console.print(f"[bold]Style Profile for {t.path}:[/bold]")
        profile = app.style_engine.build_topic_profile(t.id)

        console.print(Panel.fit(
            f"[bold]Sample Count:[/bold] {profile['sample_count']}\n"
            f"[bold]Updated:[/bold] {profile['updated_at']}",
            title=f"Style Profile: {t.name}"
        ))

        # User style
        user_style = profile['user_style']
        console.print("[bold]User Style:[/bold]")
        console.print(f"  Tone: {user_style['tone']}")
        console.print(f"  Formality: {user_style['formality']:.2f}")
        console.print(f"  Fillers: {', '.join(user_style['fillers'][:10]) or 'none'}")
        console.print(f"  Hesitations: {', '.join(user_style['hesitation_markers'][:10]) or 'none'}")
        console.print(f"  Avg Sentence Length: {user_style['avg_sentence_length']:.1f}")
        console.print(f"  Vocabulary Richness: {user_style['vocabulary_richness']:.2f}")

        # Assistant style
        assistant_style = profile['assistant_style']
        console.print("\n[bold]Assistant Style:[/bold]")
        console.print(f"  Tone: {assistant_style['tone']}")
        console.print(f"  Formality: {assistant_style['formality']:.2f}")
        console.print(f"  Fillers: {', '.join(assistant_style['fillers'][:10]) or 'none'}")
        console.print(f"  Hesitations: {', '.join(assistant_style['hesitation_markers'][:10]) or 'none'}")
        console.print(f"  Avg Sentence Length: {assistant_style['avg_sentence_length']:.1f}")
        console.print(f"  Vocabulary Richness: {assistant_style['vocabulary_richness']:.2f}")


@cli.command()
@click.argument('memory_id', type=int)
@click.option('--target-style', type=click.Choice(['user', 'assistant', 'combined']), default='combined', help='Target style to apply')
@click.pass_obj
def reconstruct_style(app: ArtificialMemoryCLI, memory_id: int, target_style: str):
    """Reconstruct memory text with a specific conversation style."""
    memory = app.store.get_memory(memory_id)
    if not memory:
        console.print(f"[red]Memory {memory_id} not found[/red]")
        return

    # Get style profile for the topic
    profile = app.style_engine.build_topic_profile(memory.topic_id)

    if target_style == 'user':
        target = ConversationStyle.from_dict(profile['user_style'])
    elif target_style == 'assistant':
        target = ConversationStyle.from_dict(profile['assistant_style'])
    else:
        target = ConversationStyle.from_dict(profile['combined_style'])

    reconstructed = app.style_engine.extractor.merge_styles(
        app.style_engine.extractor.extract_style(memory.content),
        target,
        0.3  # Keep 30% original, 70% target style
    )

    # Reconstruct text (simplified)
    reconstructed = app.style_engine.reconstruct_style(memory.content, target)

    console.print(Panel(memory.content[:500] + ("..." if len(memory.content) > 500 else ""), title="Original"))
    console.print(Panel(reconstructed[:500] + ("..." if len(reconstructed) > 500 else ""), title=f"Reconstructed ({target_style})"))

    # Show style differences
    console.print("\n[bold]Style Differences:[/bold]")
    console.print(f"  Tone: {memory.content[:50]}... → {target.tone}")
    console.print(f"  Formality: {memory.content[:50]}... → {target.formality:.2f}")
    console.print(f"  Fillers added: {', '.join(target.fillers[:5]) or 'none'}")


@cli.command()
@click.argument('query')
@click.option('--topic', help='Topic path')
@click.option('--max-tokens', default=4000, help='Max tokens for response')
@click.pass_obj
def human_recall(app: ArtificialMemoryCLI, query: str, topic: str | None, max_tokens: int):
    """Perform human-like recall with adaptive resolution."""
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topic_id = t.id
    else:
        topic_id = None

    response = app.human_recall_engine.simulate_human_recall(query, topic_id, max_tokens)
    console.print(Panel(response, title=f"Human-like Recall: {query}"))


@cli.command()
@click.argument('query')
@click.option('--topic', help='Topic path')
@click.pass_obj
def recall_explain(app: ArtificialMemoryCLI, query: str, topic: str | None):
    """Explain how human-like recall would work for a query."""
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topic_id = t.id
    else:
        topic_id = None

    explanation = app.human_recall_engine.get_recall_explanation(query, topic_id)

    console.print(Panel.fit(
        f"[bold]Query:[/bold] {explanation['query']}\n"
        f"[bold]Mode:[/bold] {explanation['mode']}\n"
        f"[bold]Resolution:[/bold] {explanation['resolution']}\n"
        f"[bold]Confidence:[/bold] {explanation['confidence']:.2f}\n"
        f"[bold]Memories Retrieved:[/bold] {explanation['memories_retrieved']}\n"
        f"[bold]Reasoning:[/bold] {explanation['reasoning']}",
        title="Human Recall Explanation"
    ))

    if explanation['adaptations']:
        table = Table(title="Resolution Adaptations")
        table.add_column("Memory ID", style="cyan")
        table.add_column("Mode", style="magenta")
        table.add_column("Target Resolution", style="yellow")
        table.add_column("Original Resolution", style="blue")
        table.add_column("Relevance", style="green")

        for a in explanation['adaptations']:
            table.add_row(
                str(a['memory_id']),
                a['mode'],
                a['target_resolution'],
                a['original_resolution'],
                f"{a['relevance']:.2f}"
            )

        console.print(table)


if __name__ == '__main__':
    cli()


# Vector Search Commands
@cli.command()
@click.argument('query')
@click.option('--topic', help='Topic path')
@click.option('--k', default=10, help='Number of results')
@click.option('--threshold', default=0.0, help='Similarity threshold')
@click.pass_obj
def vector_search(app: ArtificialMemoryCLI, query: str, topic: str | None, k: int, threshold: float):
    """Search memories using vector similarity."""
    topic_id = None
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topic_id = t.id

    results = app.vector_search_engine.search(query, topic_id=topic_id, k=k, threshold=threshold)

    console.print(Panel.fit(
        f"[bold]Query:[/bold] {query}\n"
        f"[bold]Results:[/bold] {len(results)}",
        title="Vector Search"
    ))

    if results:
        table = Table(title="Vector Search Results")
        table.add_column("Memory ID", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Preview", style="white")

        for r in results:
            mem = app.store.get_memory(r.memory_id)
            preview = mem.content[:100].replace('\n', ' ') + "..." if mem else "N/A"
            table.add_row(str(r.memory_id), f"{r.score:.4f}", preview)

        console.print(table)


@cli.command()
@click.argument('query')
@click.option('--topic', help='Topic path')
@click.option('--k', default=10, help='Number of results')
@click.option('--vector-weight', default=0.7, help='Vector search weight')
@click.option('--keyword-weight', default=0.3, help='Keyword search weight')
@click.pass_obj
def hybrid_search(app: ArtificialMemoryCLI, query: str, topic: str | None, k: int, vector_weight: float, keyword_weight: float):
    """Hybrid vector + keyword search."""
    topic_id = None
    if topic:
        t = app.conversation_manager.get_or_create_topic(topic)
        topic_id = t.id

    results = app.vector_search_engine.hybrid_search(query, topic_id=topic_id, k=k,
                                                      vector_weight=vector_weight, keyword_weight=keyword_weight)

    console.print(Panel.fit(
        f"[bold]Query:[/bold] {query}\n"
        f"[bold]Results:[/bold] {len(results)}",
        title="Hybrid Search"
    ))

    if results:
        table = Table(title="Hybrid Search Results")
        table.add_column("Memory ID", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Preview", style="white")

        for r in results:
            mem = app.store.get_memory(r.memory_id)
            preview = mem.content[:100].replace('\n', ' ') + "..." if mem else "N/A"
            table.add_row(str(r.memory_id), f"{r.score:.4f}", preview)

        console.print(table)


@cli.command()
@click.pass_obj
def vector_stats(app: ArtificialMemoryCLI):
    """Show vector index statistics."""
    stats = app.vector_search_engine.get_stats()
    console.print(Panel.fit(
        f"[bold]Total Vectors:[/bold] {stats['total_vectors']}\n"
        f"[bold]Dimension:[/bold] {stats['dimension']}\n"
        f"[bold]Index Type:[/bold] {stats['index_type']}\n"
        f"[bold]Memory Mappings:[/bold] {stats['memory_mappings']}",
        title="Vector Index Stats"
    ))


@cli.command()
@click.pass_obj
def vector_rebuild(app: ArtificialMemoryCLI):
    """Rebuild vector index from all memories."""
    console.print("[yellow]Rebuilding vector index...[/yellow]")
    app.vector_search_engine._rebuild_index()
    stats = app.vector_search_engine.get_stats()
    console.print(f"[green]Rebuilt index with {stats['total_vectors']} vectors[/green]")


if __name__ == '__main__':
    cli()
