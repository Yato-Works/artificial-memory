from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter

adapter = LongMemEvalAdapter()
items = adapter.load_dataset()

for item in items:
    if item.question_id == '0a995998':
        print('Question:', item.question)
        print('Answer:', item.answer)
        print('Answer session IDs:', item.answer_session_ids)
        print('Haystack sessions:', len(item.haystack_sessions))
        print('Haystack session IDs:', item.haystack_session_ids)
        print()
        for i, (sid, session, date) in enumerate(zip(item.haystack_session_ids, item.haystack_sessions, item.haystack_dates)):
            print('Session {}: {} ({})'.format(i, sid, date))
            for turn in session:
                print('  {}: {}...'.format(turn["role"], turn["content"][:100]))
            print()
        break