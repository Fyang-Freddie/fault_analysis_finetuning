from docx import Document
import re
import os

def is_noise_line(text: str) -> bool:
    """
    判断一行文本是否属于噪声内容，例如：
    目录行、图题、表题、图片标注、页码、小标签等
    """

    text = text.strip()

    if not text:
        return True

    # 1. 过滤很短的图片标注词
    short_noise_words = {
        "宏观", "内壁", "外壁", "中部", "整体情况",
        "位置A", "位置B", "位置C", "位置D", "位置E", "位置F",
        "启裂处", "裂纹尖端", "中间位置",
        "001VP", "002VP",
        "区域1", "区域2", "区域3", "区域4"
    }

    if text in short_noise_words:
        return True

    # 2. 过滤图题，例如：图5-1 来样管外壁宏观检查
    if re.match(r"^图\s*\d+[-－]\d+", text):
        return True

    # 3. 过滤表题，例如：表5-1 化学成分分析结果
    if re.match(r"^表\s*\d+[-－]\d+", text):
        return True

    # 4. 过滤目录标题
    if text in {"目录", "目  录", "目 录"}:
        return True

    # 5. 过滤目录中的行，例如：5.1 宏观检查    7
    if re.match(r"^\d+(\.\d+)*\s+.+\s+\d+$", text):
        return True

    # 6. 过滤纯页码
    if re.match(r"^\d+$", text):
        return True

    # 7. 过滤类似：(a)、(b)、（a）、（b）
    if re.match(r"^[\(（][a-zA-Z0-9一二三四五六七八九十]+[\)）]", text):
        return True

    # 8. 过滤只有编号或位置编号的行，例如：位置1、位置10
    if re.match(r"^位置\d+$", text):
        return True

    # 9. 过滤过短且没有中文说明意义的行
    if len(text) <= 2:
        return True

    return False


def clean_text_lines(lines: list) -> list:
    """
    对提取出来的文本行进行清洗。
    """
    cleaned = []

    for line in lines:
        line = line.strip()

        if is_noise_line(line):
            continue

        # 合并多余空格
        line = re.sub(r"\s+", " ", line)

        cleaned.append(line)

    return cleaned


def extract_docx_content(docx_path: str, extract_tables: bool = True) -> str:
    """
    提取并清洗单个 docx 文件中的文本内容。

    参数:
        docx_path: docx 文件路径
        extract_tables: 是否提取表格内容，默认提取

    返回:
        str: 清洗后的完整文本内容
    """

    doc = Document(docx_path)
    content_parts = []

    # 1. 提取普通段落
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            content_parts.append(text)

    # 2. 提取表格内容
    if extract_tables:
        for table in doc.tables:
            for row in table.rows:
                row_text = []

                for cell in row.cells:
                    cell_text = cell.text.strip()

                    if cell_text and not is_noise_line(cell_text):
                        row_text.append(cell_text)

                if row_text:
                    content_parts.append(" | ".join(row_text))

    # 3. 清洗噪声行
    cleaned_lines = clean_text_lines(content_parts)

    return "\n".join(cleaned_lines)


def main():
    input_folder = "papers_failure_analysis2"

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".docx") and not filename.startswith("~$"):
            file_path = os.path.join(input_folder, filename)

            text = extract_docx_content(file_path,extract_tables=False)

            print(f"正在处理：{filename}")
            print(text)

if __name__=="__main__":
    main()