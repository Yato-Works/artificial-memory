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
        
        # Score all records and find the answer sessions
        scored = []
        for r in all_records:
            from artificial_memory.core.ir.memory_types import ApexMemoryUnit, MemoryRole
            unit = ApexMemoryUnit(ir=r, role=MemoryRole.STATE)
            breakdown = scorer.compute_score(item.question, unit, weights)
            scored.append((breakdown.total_score, r, breakdown))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        print('Question:', item.question)
        print('\nTop 30 scored:')
        for score, r, bd in scored[:30]:
            is_ans = any(sid in r.raw_content for sid in item.answer_session_ids)
            marker = ' ANSWER' if is_ans else ''
            # Safe print
            content = r.raw_content[:120].encode('ascii', 'ignore').decode('ascii')
            print(f'  {score:.1f} L:{bd.s_lexical:.1f} A:{bd.s_actor:.1f} E:{bd.s_entity:.1f} S:{bd.s_semantic:.1f} T:{bd.s_temporal:.1f} P:{bd.s_provenance:.1f} | {content}...{marker}')
        break