from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Tô Lâm (sinh ngày 10 tháng 7 năm 1957 tại Hưng Yên) là một chính trị gia người Việt Nam, Tướng lĩnh Công an nhân dân Việt Nam, hàm Đại tướng. Ông hiện đang giữ chức vụ Tổng Bí thư Ban Chấp hành Trung ương Đảng Cộng sản Việt Nam, Bí thư Quân ủy Trung ương kể từ ngày 3 tháng 8 năm 2024. Ngoài ra, ông còn đang đảm nhiệm các chức vụ Trưởng Ban Chỉ đạo Trung ương về phòng chống tham nhũng, tiêu cực và Đại biểu Quốc hội Việt Nam khóa XV nhiệm kì 2021–2026, thuộc Đoàn Đại biểu Quốc hội Thành phố Hà Nội.[1] Trước đó, ông từng giữ chức Bộ trưởng Bộ Công an từ ngày 9 tháng 4 năm 2016 đến ngày 22 tháng 5 năm 2024 sau đó được bầu chức Chủ tịch nước Cộng hoà xã hội chủ nghĩa Việt Nam từ ngày 22 tháng 5 năm 2024 đến ngày 21 tháng 10 năm 2024 khi Lương Cường được Quốc hội bầu làm Chủ tịch nước Cộng hoà xã hội chủ nghĩa Việt Nam.
Tô Lâm là Đảng viên từ năm 1981. Ông có học hàm Giáo sư Khoa học An ninh, học vị Tiến sĩ Luật học, Cao cấp lí luận chính trị, cấp hàm Đại tướng Công an nhân dân Việt Nam, là Ủy viên Bộ Chính trị khóa XII, XIII, Ủy viên Ban Chấp hành Trung ương Đảng khóa XI, XII, XIII, từng giữ các chức vụ Bí thư Đảng ủy Công an Trung ương, Bộ trưởng Bộ Công an, Thứ trưởng Bộ Công an và Tổng cục trưởng Tổng cục An ninh I.[2] Ông được cho là đã lãnh đạo chiến dịch chống lại những người bất đồng chính kiến, đàn áp các tổ chức xã hội dân sự, thắt chặt kiểm soát internet và lên kế hoạch truy bắt và dẫn độ Trịnh Xuân Thanh.[3][4][5]
Ngày 22 tháng 5 năm 2024, sau khi Võ Văn Thưởng từ chức 2 tháng trước đó, Tô Lâm được Quốc hội Việt Nam khóa XV bầu làm Chủ tịch nước.[6] Chỉ chưa đầy 3 tháng sau, vào ngày 3 tháng 8 năm 2024, ông được Ban Chấp hành Trung ương Đảng khoá XIII bầu làm Tổng Bí thư tại Hội nghị Trung ương bất thường khóa XIII, kế nhiệm Tổng Bí thư Nguyễn Phú Trọng vừa từ trần 2 tuần trước đó.[7] Việc này đưa Tô Lâm trở thành người thứ tư trong lịch sử chính trị Việt Nam giữ đồng lúc cả chức vụ đứng đầu Đảng và đứng đầu Nhà nước trong cùng một khoảng thời gian, sau Hồ Chí Minh, Trường Chinh và Nguyễn Phú Trọng. Ông đã kiêm nhiệm chức Chủ tịch nước đến tháng 10 cùng năm, khi Quốc hội bầu Lương Cường lên thay thế để đảm bảo nguyên tắc lãnh đạo tập thể của Đảng là không nhất thể hóa 2 chức danh cao nhất cho 1 người nắm giữ, khiến ông trở thành Chủ tịch nước có nhiệm kỳ ngắn nhất Việt Nam chỉ sau 5 tháng cầm quyền. Ông được cho là đã tiếp tục Chiến dịch đốt lò do người tiền nhiệm của ông là cố Tổng Bí thư Nguyễn Phú Trọng khởi xướng.
"""
    summary_template = """
    given the information {information}, generate a concise summary in Vietnamese.
    1. a short summary (max 3 sentences)
    2. list of 3 key points about the information
    3. three interesting facts related to the information
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(model_name="gpt-5", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print("Summary Response:")
    print(response.content)

if __name__ == "__main__":
    main()
