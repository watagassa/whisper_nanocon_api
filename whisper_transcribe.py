import whisper

def main():
    model = whisper.load_model("base")
    file_path = input("音声ファイル(mp3/mp4)のパスを入力してください: ")

    try:
        # return_segments は削除
        result = model.transcribe(file_path, language="ja")

        print("\n--- 認識結果 ---")
        print(result["text"])

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
