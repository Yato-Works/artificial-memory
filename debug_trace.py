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
        
        # Let's trace through the MSC compilation step by step
        from artificial_memory.recall.state_reconstructor import StateReconstructor
        reconstructor = StateReconstructor()
        intent, candidate_units = reconstructor.reconstruct_world(item.question, all_records)
        print('Intent:', intent)
        print('Candidate units:', len(candidate_units))
        for u in candidate_units[:10]:
            print('  ', u.ir.raw_content[:120])
        print()
        
        # Check the adaptive search
        from artificial_memory.recall.adaptive_search import AdaptiveEvidenceSearcher
        from artificial_memory.recall.query_planner import QueryPlanner
        from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph
        
        planner = QueryPlanner()
        plan = planner.plan(item.question)
        print('Plan:', plan)
        
        graph = UnifiedPropositionGraph()
        graph.build_from_records(all_records)
        print('Graph propositions:', len(graph.propositions))
        
        searcher = AdaptiveEvidenceSearcher()
        search_res = searcher.search(plan, graph, candidate_units)
        print('Search results:', len(search_res.selected_propositions))
        for p in search_res.selected_propositions:
            print('  ', p.raw_text[:120])
        print('Grounding tags:', search_res.grounding_tags)
        print('Hops:', search_res.hops_traversed)
        print('Sufficient:', search_res.is_sufficient)
        break