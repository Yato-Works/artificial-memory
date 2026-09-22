from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.recall.state_reconstructor import StateReconstructor

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()

for item in items:
    if item.question_id == 'gpt4_d84a3211':
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
        
        reconstructor = StateReconstructor()
        intent, candidate_units = reconstructor.reconstruct_world(item.question, all_records)
        
        # Check answer sessions for keywords
        keywords = ["spent", "cost", "paid", "price", "$", "dollar", "expense", "bought", "purchased"]
        
        for i, u in enumerate(candidate_units):
            is_ans = any(ans_sid in u.ir.raw_content for ans_sid in item.answer_session_ids)
            if is_ans:
                content_lower = u.ir.raw_content.lower()
                has_kw = any(kw in content_lower for kw in keywords)
                has_num = bool(__import__('re').search(r"\b\d+\b", content_lower)) or "$" in content_lower
                if has_kw and has_num:
                    m = __import__('re').search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", u.ir.raw_content)
                    sid = m.group(1) if m else "unknown"
                    print(f'  [{i}] ANSWER sid={sid}: has_kw={has_kw}, has_num={has_num}')
                    # Show the matching parts
                    for kw in keywords:
                        if kw in content_lower:
                            idx = content_lower.index(kw)
                            print(f'    Found "{kw}" at {idx}: ...{u.ir.raw_content[max(0,idx-30):idx+30]}...')
        break