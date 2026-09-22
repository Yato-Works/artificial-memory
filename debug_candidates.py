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
        
        # Check state reconstructor
        from artificial_memory.recall.state_reconstructor import StateReconstructor
        reconstructor = StateReconstructor()
        intent, candidate_units = reconstructor.reconstruct_world(item.question, all_records)
        print('Intent:', intent)
        print('Candidate units:', len(candidate_units))
        
        # Show top 20 with session IDs
        import re
        for i, u in enumerate(candidate_units[:20]):
            m = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", u.ir.raw_content)
            sid = m.group(1) if m else "unknown"
            is_ans = any(ans_sid in u.ir.raw_content for ans_sid in item.answer_session_ids)
            marker = ' ANSWER' if is_ans else ''
            content = u.ir.raw_content[:100].encode('ascii', 'ignore').decode('ascii')
            print(f'  {i}: [{sid}] {content}...{marker}')
        break