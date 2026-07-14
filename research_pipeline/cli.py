"""CLI entrypoint for all multi-agent pipelines."""
from __future__ import annotations

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run any of the implemented multi-agent pipelines")
    parser.add_argument("--pipeline",
                        choices=["rag", "active-learning", "customer-intelligence"],
                        required=True)
    parser.add_argument("--topic",    default="Advancements in AI in the medical field")
    parser.add_argument("--question", default="How does AI help in medicine?")
    parser.add_argument("--message",  default="I need a refund for my last invoice")
    parser.add_argument("--session",  default="cli-session")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--db-path", default="output/vector_db.pkl",
                        help="Path to persist the RAG vector DB state across CLI runs.")
    # Active Learning configuration (can be overridden via env vars AL_*)
    parser.add_argument("--al-token-budget", type=int, default=None,
                        help="Token budget for Active Learning (env: AL_TOKEN_BUDGET)")
    parser.add_argument("--al-target-accuracy", type=float, default=None,
                        help="Target accuracy for Active Learning trainer (env: AL_TARGET_ACCURACY)")
    parser.add_argument("--al-batch-size", type=int, default=None,
                        help="Annotation batch size (env: AL_BATCH_SIZE)")
    parser.add_argument("--al-num-samples", type=int, default=None,
                        help="Number of synthetic samples to generate (env: AL_NUM_SAMPLES)")
    parser.add_argument("--al-annotation-confidence-target", type=float, default=None,
                        help="Annotation confidence stopping target (env: AL_ANNOTATION_CONFIDENCE_TARGET)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--al-use-real-dataset", dest="al_use_real_dataset", action="store_true",
                       help="Use AG News real dataset for Active Learning if available (default)")
    group.add_argument("--no-al-use-real-dataset", dest="al_use_real_dataset", action="store_false",
                       help="Force synthetic dataset even if AG News is available")
    parser.set_defaults(al_use_real_dataset=True)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # RAG pipeline
    # -----------------------------------------------------------------------
    if args.pipeline == "rag":
        from research_pipeline.rag.ingestion   import run_ingestion_pipeline
        from research_pipeline.rag.vector_db   import VectorDB
        from research_pipeline.rag.conversation import ConversationalRAGAgent

        print("\n=== Agentic RAG Pipeline ===")
        db = VectorDB(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            storage_path=args.db_path,
        )
        report = run_ingestion_pipeline(
            topic=args.topic,
            db=db,
            max_sources=args.max_sources,
            retention_days=args.retention_days,
            token_budget=args.token_budget,
        )
        print(f"Ingestion complete: {report['new_chunks_added']} new chunks, "
              f"{report['duplicates_skipped']} duplicates skipped.")

        agent  = ConversationalRAGAgent(vector_db=db)
        result = agent.answer_query(session_id=args.session, query=args.question)

        print(f"\nQuery: {args.question}")
        print(f"Grounded: {result['grounded']}")
        print(f"\nAnswer:\n{result['response']}")
        if result["sources"]:
            print("\nSources cited:")
            for s in result["sources"]:
                print(f"  [{s['id']}] {s['title']} — {s['url']}")

    # -----------------------------------------------------------------------
    # Active Learning pipeline
    # -----------------------------------------------------------------------
    elif args.pipeline == "active-learning":
        from research_pipeline.active_learning.annotation_pipeline import ActiveLearningPipeline
        from research_pipeline.active_learning.config import load_from_args as load_al_config

        print("\n=== Active Learning Pipeline ===")
        al_cfg = load_al_config(args)
        pipeline = ActiveLearningPipeline(
            token_budget=al_cfg.token_budget,
            target_accuracy=al_cfg.target_accuracy,
            annotation_confidence_target=al_cfg.annotation_confidence_target,
            batch_size=al_cfg.batch_size,
            num_samples=al_cfg.num_samples,
            use_real_dataset=bool(args.al_use_real_dataset),
        )
        results = pipeline.run()

        print(f"\nDataset          : {'AG News (real)' if results['used_real_dataset'] else 'synthetic fallback'}")
        print(f"Annotator LLM    : {'live model calls' if results['used_real_llm'] else 'offline keyword fallback (no API key set)'}")
        print(f"\nTotal iterations : {results['total_iterations']}")
        print(f"Labeled samples  : {results['labeled_count']}")
        print(f"Tokens used      : {results['tokens_used']}")
        print(f"LLM calls        : {results['llm_calls']}  (fallback calls: {results['fallback_calls']})")
        print(f"Mean confidence  : {results['mean_confidence']:.4f}")
        print(f"Stop reason      : {results['stop_reason']}")
        print(f"\nFinal model      : {results['final_report']['model_name']}")
        print(f"Test accuracy    : {results['final_report']['test_accuracy']:.4f}")
        print("\nPer-class metrics:")
        for cls, m in results["final_report"]["metrics_per_class"].items():
            print(f"  {cls:<20} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1_score']:.3f}")

    # -----------------------------------------------------------------------
    # Customer Intelligence pipeline
    # -----------------------------------------------------------------------
    else:
        from research_pipeline.customer_intelligence.pipeline import (
            CustomerIntelligencePipeline, CustomerTurn)
        from research_pipeline.customer_intelligence import lmcache_production

        print("\n=== Customer Intelligence Pipeline ===")
        if lmcache_production.is_real_lmcache_available():
            print("Real LMCache + vLLM backend detected (GPU) — see lmcache_production.py")
        else:
            print("No GPU/lmcache/vllm detected — using the portable KV-cache simulation "
                  "(see KVCacheManager in pipeline.py). Real GPU integration: lmcache_production.py")
        pipeline = CustomerIntelligencePipeline()

        # Simulate a multi-turn conversation
        messages = [
            args.message,
            "Can you tell me more about the pricing plans?",
            "What is the SLA for the Enterprise tier?",
        ]
        for msg in messages:
            turn   = CustomerTurn(user_message=msg, session_id=args.session)
            result = pipeline.handle_turn(turn)
            print(f"\nUser   : {msg}")
            print(f"Intent : {result['intent']} (conf={result['confidence']:.2f})")
            print(f"Quality: {result['quality_score']:.3f}  approved={result['approved']}")
            print(f"Agent  : {result['response'][:200]}...")
            print(f"KV cache hit rate: {result['kv_cache']['cache_hit_rate']*100:.1f}%  "
                  f"tokens saved: {result['kv_cache']['total_tokens_saved']}")


if __name__ == "__main__":
    main()
