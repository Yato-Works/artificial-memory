from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
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
        
        # Check the query planner
        from artificial_memory.recall.query_planner import QueryPlanner
        planner = QueryPlanner()
        plan = planner.plan(item.question)
        print('Plan:')
        print('  raw_query:', plan.raw_query)
        print('  intent:', plan.intent)
        print('  target_entities:', plan.target_entities)
        print('  missing_slots:', plan.missing_slots)
        print()
        
        # Check state reconstructor intent
        from artificial_memory.recall.state_reconstructor import StateReconstructor
        reconstructor = StateReconstructor()
        intent = reconstructor.classify_intent(item.question)
        print('StateReconstructor intent:', intent)
        
        # Check evidence keywords
        q_lower = item.question.lower()
        evidence_kw = [
            "did alice propose", "did bob propose", "did the user propose",
            "who proposed", "who suggested", "at any point", "did someone suggest",
            "was there a proposal", "what did alice say", "what did bob say",
        ]
        for kw in evidence_kw:
            if kw in q_lower:
                print(f'  Evidence keyword match: {kw}')
        
        # Check if "how many" should trigger aggregation
        print('Question:', item.question)
        print('Has "how many":', 'how many' in q_lower)
        print('Has "total":', 'total' in q_lower)
        
        # Check session fuser
        from artificial_memory.protein.session_fuser import SessionFuser
        fuser = SessionFuser()
        print('Is aggregation query:', fuser.is_aggregation_query(item.question))
        print('Determined unit:', fuser.determine_unit(item.question))
        break