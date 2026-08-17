import os
from pathlib import Path

from text_scatter import PDFExtract


def main():

    pdf_file = Path(
        r"D:\pycharmcode\ElectricityLLM\LLM-data\Document\MATLAB_Simulink电力电子仿真操作技术文档.pdf"
    )

    output_file = Path(r"D:\pycharmcode\ElectricityLLM\LLM-data\json_data\simulink.json")

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_file}")

    extractor = PDFExtract(
        pdf_file_address=pdf_file,
        cache_address=Path(r"D:\pycharmcode\ElectricityLLM\ElectricityLLM\Document\temp\infineon"),
        bucket_name="electricity-document",
        title=pdf_file.stem,
        minio_endpoint="127.0.0.1:9000",
        minio_access_key=os.environ["MINIO_ACCESS_KEY"],
        minio_secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False
    )

    extractor.primary_deal()
    extractor.pdf_picture_clear()
    extractor.remove_visual_content()

    chunks = extractor.run_text_pipeline(
        output_file=output_file,
        max_chars=800,
        exclude_appendix=True,
        overlap_sentences=1,
        generate_descriptions=True,
        generate_embedding=True,
        description_min_chars=15,
        checkpoint_interval=1
    )

    print(f"切分完成，共生成 {len(chunks)} 个片段。")
    print(f"JSON 文件：{output_file}")


if __name__ == "__main__":
    main()

