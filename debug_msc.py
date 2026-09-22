from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.protein.session_fuser import SessionFuser
import re

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()

for item in items:
    if item.question_id == 'gpt4_d84a3211':
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
        
        # Trace selection with second pass
        is_aggregation = intent.name == 'AGGREGATION_QUERY'
        selected_units = []
        curr_tokens = 0
        max_units = 10 if is_aggregation else 6
        target_token_budget = 180
        cert = compiler.checker.check(item.question, intent, selected_units)
        seen_sessions = set()
        SESSION_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]")
        
        print('\n=== INITIAL SELECTION ===')
        for i, u in enumerate(candidate_units[:max_units]):
            u_tok = len(u.ir.raw_content.split())
            m = SESSION_PATTERN.search(u.ir.raw_content)
            sid = m.group(1) if m else "unknown"
            
            if selected_units and (curr_tokens + u_tok > target_token_budget):
                if is_aggregation:
                    print(f'  Candidate {i}: sid={sid}, tokens={u_tok} -> SKIP (budget)')
                    continue
                break
            
            session_count = sum(1 for su in selected_units 
                               if SESSION_PATTERN.search(su.ir.raw_content) and SESSION_PATTERN.search(su.ir.raw_content).group(1) == sid)
            if session_count >= 2 and len(seen_sessions) < 4:
                print(f'  Candidate {i}: sid={sid}, tokens={u_tok} -> SKIP (session_count={session_count})')
                continue
            
            selected_units.append(u)
            curr_tokens += u_tok
            if m:
                seen_sessions.add(sid)
            
            cert = compiler.checker.check(item.question, intent, selected_units)
            print(f'  Candidate {i}: sid={sid}, tokens={u_tok} -> ADDED (cert.sufficient={cert.is_sufficient}, total_tokens={curr_tokens}, sessions={len(seen_sessions)})')
            
            if cert.is_sufficient and len(selected_units) >= 4 and curr_tokens >= 110:
                print('  -> BREAK: sufficient')
                break
        
        print(f'\nAfter initial: selected={len(selected_units)}, sessions={len(seen_sessions)}, tokens={curr_tokens}')
        
        # Recovery
        print('\n=== RECOVERY ===')
        if not cert.is_sufficient and len(candidate_units) > len(selected_units):
            restored_count = 0
            for extra in candidate_units[len(selected_units):8]:
                u_tok = len(extra.ir.raw_content.split())
                if curr_tokens + u_tok > target_token_budget + 40:
                    break
                selected_units.append(extra)
                curr_tokens += u_tok
                restored_count += 1
                new_cert = compiler.checker.check(item.question, intent, selected_units)
                if new_cert.is_sufficient:
                    cert = new_cert
                    cert.omitted_records_restored = restored_count
                    break
        
        print(f'After recovery: selected={len(selected_units)}, sessions={len(seen_sessions)}, tokens={curr_tokens}')
        
        # Second pass
        print('\n=== SECOND PASS ===')
        if is_aggregation and len(candidate_units) > len(selected_units):
            fuser = SessionFuser()
            unit = fuser.determine_unit(item.question)
            print(f'Unit: {unit}')
            
            agg_evidence_keywords = {
                "$": ["spent", "cost", "paid", "price", "$", "dollar", "expense", "bought", "purchased"],
                "items of clothing": ["pick up", "return", "exchange", "bought", "got", "blazer", "boots", "jeans", "shirt"],
                "doctors": ["doctor", "dr.", "dermatologist", "physician", "specialist", "ent"],
                "plants": ["plant", "lily", "succulent", "fern", "basil", "nursery", "bought", "acquired"],
                "projects": ["project", "lead", "leading", "led", "completed", "manage", "launch"],
                "days": ["day", "days", "camping", "trip", "visit", "spent"],
                "weeks": ["week", "weeks", "watch", "marvel", "movie", "film"],
                "hours": ["hour", "hours", "jog", "run", "exercise", "workout"],
                "items": ["item", "items", "count", "total", "how many", "how much"],
            }
            keywords = agg_evidence_keywords.get(unit, agg_evidence_keywords["items"])
            print(f'Keywords: {keywords}')
            
            seen_sessions = set()
            for u in selected_units:
                m = SESSION_PATTERN.search(u.ir.raw_content)
                if m:
                    seen_sessions.add(m.group(1))
            print(f'Current sessions: {seen_sessions}')
            
            added = 0
            for i, extra in enumerate(candidate_units):
                if len(selected_units) >= max_units:
                    print(f'  Stop: max_units reached')
                    break
                if extra in selected_units:
                    continue
                u_tok = len(extra.ir.raw_content.split())
                if curr_tokens + u_tok > target_token_budget + 60:
                    print(f'  Candidate {i}: tokens={u_tok}, budget exceeded')
                    continue
                
                content_lower = extra.ir.raw_content.lower()
                has_evidence = any(kw in content_lower for kw in keywords)
                has_numbers = bool(re.search(r"\b\d+\b", content_lower)) or "$" in content_lower
                
                m = SESSION_PATTERN.search(extra.ir.raw_content)
                sid = m.group(1) if m else "unknown"
                
                if has_evidence and has_numbers and m and sid not in seen_sessions:
                    selected_units.append(extra)
                    curr_tokens += u_tok
                    seen_sessions.add(sid)
                    added += 1
                    cert = compiler.checker.check(item.question, intent, selected_units)
                    print(f'  Candidate {i}: sid={sid}, ADDED (evidence={has_evidence}, numbers={has_numbers}) -> cert.sufficient={cert.is_sufficient}')
            
            print(f'Added {added} units in second pass')
        
        print(f'\nFinal: selected={len(selected_units)}, sessions={len(seen_sessions)}, tokens={curr_tokens}')
        for u in selected_units:
            m = SESSION_PATTERN.search(u.ir.raw_content)
            sid = m.group(1) if m else "unknown"
            content = u.ir.raw_content[:100].encode('ascii', 'ignore').decode('ascii')
            print(f'  [{sid}] {content}...')
        break