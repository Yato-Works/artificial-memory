from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()

for item in items:
    if item.question_id == '0a995998':
        # Ingest haystack sessions
        all_records = []
        for s_idx, session in enumerate(item.haystack_sessions):
            s_date = item.haystack_dates[s_idx] if s_idx < len(item.haystack_dates) else ''
            sid = item.haystack_session_ids[s_idx] if s_idx < len(item.haystack_session_ids) else ''
            for turn in session:
                speaker = turn.get('role', 'user')
                content = turn.get('content', '')
                recs = adapter.extractor.extract(content, default_source=speaker)
                for r in recs:
                    r.raw_content = f'[{sid} on {s_date}] {speaker}: {content}' if s_date else f'[{sid}] {speaker}: {content}'
                    r.time_scope = s_date
                    all_records.append(r)
        
        # Compile MSC with tracing
        compiler = adapter.compiler
        
        # Trace the compilation
        from artificial_memory.recall.query_planner import QueryPlanner
        from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph
        from artificial_memory.recall.state_reconstructor import StateReconstructor
        from artificial_memory.recall.adaptive_search import AdaptiveEvidenceSearcher
        from artificial_memory.recall.proposition_integrity_gate import PropositionIntegrityGate
        from artificial_memory.recall.state_supersession_engine import StateSupersessionEngine
        from artificial_memory.memory.persona_store import PersonaStore
        from artificial_memory.recall.temporal_resolver import TemporalResolver
        
        plan = compiler.planner.plan(item.question)
        graph = UnifiedPropositionGraph()
        graph.build_from_records(all_records)
        
        temporal_grounding = compiler.temporal_resolver.resolve(
            item.question,
            all_records,
            reference_date_str=item.question_date,
        )
        
        compiler.persona_store.attributes.clear()
        compiler.persona_store.ingest_records(all_records)
        persona_grounding = compiler.persona_store.get_persona_grounding(item.question)
        
        state_resolution = compiler.state_engine.resolve(plan, graph)
        
        intent, candidate_units = compiler.reconstructor.reconstruct_world(item.question, all_records)
        
        search_res = compiler.adaptive_searcher.search(plan, graph, candidate_units)
        
        integrity = compiler.integrity_gate.check(plan, search_res.selected_propositions)
        
        print('Intent:', intent)
        print('Candidate units:', len(candidate_units))
        
        # Trace selection
        selected_units = []
        curr_tokens = 0
        max_units = 10
        cert = compiler.checker.check(item.question, intent, selected_units)
        seen_sessions = set()
        import re
        SESSION_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]")
        
        for i, u in enumerate(candidate_units[:max_units]):
            u_tok = len(u.ir.raw_content.split())
            m = SESSION_PATTERN.search(u.ir.raw_content)
            sid = m.group(1) if m else "unknown"
            
            print(f'  Candidate {i}: sid={sid}, tokens={u_tok}, curr_tokens={curr_tokens}')
            
            if selected_units and (curr_tokens + u_tok > 180):
                print('    -> SKIP: token budget exceeded')
                break
            
            session_count = sum(1 for su in selected_units 
                               if SESSION_PATTERN.search(su.ir.raw_content) and SESSION_PATTERN.search(su.ir.raw_content).group(1) == sid)
            if session_count >= 2 and len(seen_sessions) < 4:
                print(f'    -> SKIP: session_count={session_count}, seen_sessions={len(seen_sessions)}')
                continue
            
            selected_units.append(u)
            curr_tokens += u_tok
            if m:
                seen_sessions.add(sid)
            
            cert = compiler.checker.check(item.question, intent, selected_units)
            print(f'    -> ADDED: cert.is_sufficient={cert.is_sufficient}, selected={len(selected_units)}, sessions={len(seen_sessions)}')
            
            if cert.is_sufficient and len(selected_units) >= 4 and curr_tokens >= 110:
                print('    -> BREAK: sufficient')
                break
        
        print('\nFinal selected:')
        for u in selected_units:
            m = SESSION_PATTERN.search(u.ir.raw_content)
            sid = m.group(1) if m else "unknown"
            content = u.ir.raw_content[:80].encode('ascii', 'ignore').decode('ascii')
            print(f'  [{sid}] {content}...')
        
        print('\nFinal cert:', cert.is_sufficient, cert.entity_coverage, cert.property_coverage, cert.temporal_coverage, cert.conflict_coverage, cert.decision_coverage)
        break