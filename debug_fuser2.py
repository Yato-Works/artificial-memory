from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()

# Test a failing multi-session question
for item in items:
    if item.question_id == 'gpt4_d84a3211':  # "How many days did I spend on camping trips"
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
        
        # Compile MSC
        pcc = adapter.compiler.compile(item.question, all_records, reference_date_str=item.question_date)
        
        # Check session fuser
        fuser = adapter.compiler.session_fuser
        agg_res = fuser.fuse(item.question, pcc.context_text)
        print('Question:', item.question)
        print('GT:', item.answer)
        print('Fuser result:')
        print('  is_aggregation_query:', agg_res.is_aggregation_query)
        print('  total_value:', agg_res.total_value)
        print('  unit:', agg_res.unit)
        print('  found_snippets:', agg_res.found_snippets)
        print('  certificate:', agg_res.certificate)
        
        print('\nContext:')
        content = pcc.context_text.encode('ascii', 'ignore').decode('ascii')
        print(content[:2000])
        break