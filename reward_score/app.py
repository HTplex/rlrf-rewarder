

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
from svg_rewarder import RewarderV2
import uvicorn


app = FastAPI(
    title="Reward Score API",
    description=(
        "Return reward score for a given image and text."
    ),
    version="0.3.0",
)

all_rewarders = {
}

@app.post(
    "/score_text_svg",
    response_class=JSONResponse,
)
def score_text_svg(payload: dict):
    try:
        config = payload["config"]
        if config not in all_rewarders:
            config_path = os.path.join(os.path.dirname(__file__), "configs", config + ".yaml")
            all_rewarders[config] = RewarderV2(config_path=config_path)
        
        return all_rewarders[config].score_text_svg(payload["text_gt"], payload["svg_pred"])
        
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        raise HTTPException(status_code=500, detail=traceback_str)
    
if __name__ == "__main__":
    """
    Launch the FastAPI service with:

        $ python scripts/app.py

    Requests must include:

        X-API-Key: <your-api-key>
    """
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8200)),
        log_level="info",
    )