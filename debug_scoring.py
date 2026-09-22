from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

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
        
        # Check scoring for each record
        from artificial_memory.recall.state_reconstructor import StateReconstructor
        from artificial_memory.recall.evidence_scorer import UniversalEvidenceScorer, EvidenceScoreWeights
        
        reconstructor = StateReconstructor()
        scorer = UniversalEvidenceScorer()
        weights = EvidenceScoreWeights()
        
        q_lower = item.question.lower()
        target_entity = reconstructor._extract_target_entity(item.question)
        target_property = reconstructor._extract_target_property(item.question)
        target_date = reconstructor._extract_target_date(item.question)
        target_actor = reconstructor._extract_target_actor(item.question)
        
        print('Question:', item.question)
        print('target_entity:', target_entity)
        print('target_property:', target_property)
        print('target_date:', target_date)
        print('target_actor:', target_actor)
        print()
        
        # Score all records and find the answer sessions
        scored = []
        for r in all_records:
            from artificial_memory.core.ir.memory_types import ApexMemoryUnit, MemoryRole
            unit = ApexMemoryUnit(ir=r, role=MemoryRole.STATE)
            breakdown = scorer.compute_score(item.question, unit, weights)
            scored.append((breakdown.total_score, r))
            
            # Check if this is an answer session
            if any(sid in r.raw_content for sid in item.answer_session_ids):
                print(f'ANSWER SESSION: Score={breakdown.total_score:.1f} | {r.raw_content[:150]}...')
        
        scored.sort(key=lambda x: x[0], reverse=True)
        print('\nTop 20 scored:')
        for score, r in scored[:20]:
            is_ans = any(sid in r.raw_content for sid in item.answer_session_ids)
            marker = ' >>> ANSWER' if is_ans else ''
            print(f'  {score:.1f} | {r.raw_content[:120]}...{marker}')
        break