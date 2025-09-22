import asyncio
from app.core.config import settings
from app.services.web_search import web_search_service
from app.services.memo_generator import MemoGeneratorService

async def main():
    print("=== Env & Config Check ===")
    print("Search enabled:", getattr(web_search_service, 'search_enabled', None))
    print("Search engine ID (cx):", getattr(web_search_service, 'google_search_engine_id', ''))
    print("Vertex model:", settings.VERTEX_AI_MODEL)
    print()

    if getattr(web_search_service, 'search_enabled', False):
        print("=== Direct Web Search Test ===")
        data = await web_search_service.search_executive_bio("Tim Cook", "Apple")
        print("Raw result batches:", len(data.get("raw_results", [])))
        print(web_search_service.format_bio_section(data))
        print()

        print("=== SFE Bio via Memo Generator ===")
        mg = MemoGeneratorService()
        out = await mg._generate_sfe_bio_with_web_search({
            "Company Name": "Apple Inc.",
            "Company Location": "Cupertino, CA",
            "Leadership/Executive Team": "Tim Cook (CEO)"
        })
        print(out)
    else:
        print("Web search disabled (missing GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID).")

if __name__ == "__main__":
    asyncio.run(main())
