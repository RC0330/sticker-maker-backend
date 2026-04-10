from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 💡 關鍵改變：先把大腦設為「空」，不要一開始就下載
ai_session = None

@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...), post_processing: float = Form(0.5)):
    global ai_session # 宣告要使用外面的變數
    
    try:
        # 💡 只有在「第一次有人傳圖片來」的時候，才去下載大腦
        if ai_session is None:
            print("首次去背：正在載入 AI 模型 (u2netp)...請稍候")
            ai_session = new_session("u2netp")
            
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data))

        print("正在執行去背運算...")
        
        output_image = remove(
            input_image,
            session=ai_session, # 使用剛才準備好的大腦
            alpha_matting=False
        )
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        print("🎉 運算完成！")
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"\n!!! 發生錯誤：{e}")
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    import os
    # 讓 Render 動態決定門牌號碼
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
