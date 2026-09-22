from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()

for item in items:
    if item.question_id == 'gpt4_d84a3211':
        print('Question:', item.question)
        print('Answer:', item.answer)
        print('Answer session IDs:', item.answer_session_ids)
        print()
        for i, (sid, session, date) in enumerate(zip(item.haystack_session_ids, item.haystack_sessions, item.haystack_dates)):
            if sid in item.answer_session_ids:
                print('ANSWER Session {}: {} ({})'.format(i, sid, date))
                for turn in session:
                    content = turn["content"].encode('ascii', 'ignore').decode('ascii')
                    print('  {}: {}...'.format(turn["role"], content[:200]))
                print()
        break