# COMPARE WITH FRIEND — TRUST Agent V10 Benchmark

> Bảng 20 test case "bẫy" từ ViFactCheck. V10 đã có kết quả. Competitor tự điền.

---

## How to use this table

1. Run your system on the same 20 `statement` values
2. Fill in `Competitor_A` columns with your verdict and correctness
3. Compare accuracy at the bottom summary row
4. Pay attention to rows marked **"BẪY"** — these are the trick cases

---

## 20 Bẫy Test Cases

| # | ID | Expected | Statement | V10 Verdict | V10 OK? | Competitor_A Verdict | Competitor_A OK? | Competitor_B Verdict | Competitor_B OK? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1404 | REAL | Thủ tướng đã ký quyết định cho Chủ tịch Tập đoàn Điện lực Việt Nam (EVN) nghỉ hưu từ 1/5. | UNCERTAIN | ✗ | | | | |
| 2 | 1449 | REAL | Các cơ quan tài chính toàn cầu đang giám sát chặt chẽ hệ thống ngân hàng để ngăn chặn nguy cơ khủng hoảng tín dụng. | REAL | ✓ | | | | |
| 3 | 6147 | REAL | Dây hãm là phần mô có tính đàn hồi nằm dưới bao quy đầu và chứa nhiều dây thần kinh, mạch máu. | REAL | ✓ | | | | |
| 4 | 3379 | REAL | Oh Sehun là nam thần tượng nổi tiếng của Hàn Quốc, ra mắt khán giả vào năm 2012 với tư cách là thành viên nhóm nhạc nam EXO, sau đó EXO trở thành nhóm nhạc nam hàng đầu Kpop vào năm 2013 và duy trì độ nổi tiếng đến nay. | REAL | ✓ | | | | |
| 5 ⚠️ | 1172 | **FAKE** BẪY | Mbappe ở tuổi 24 đã tham gia 66 trận đấu quốc tế, ghi 40 bàn, ghi bàn và giành chiến thắng cho Les Bleus trước Hà Lan là cách duy nhất để ngôi sao của PSG dẹp tan những tranh luận xung quanh vai trò mới của anh. | REAL | ✗ | | | | |
| 6 | 3313 | REAL | Trần Thị Tường Anh vinh dự là học sinh Việt Nam được nhận học bổng Freeman Asian Scholarship của Wesleyan University ở Mỹ trị giá lên đến 370.000 USD. | REAL | ✓ | | | | |
| 7 | 5974 | REAL | TS Nguyễn Kim Quốc cho biết 100% phòng thi đều được trang bị đầy đủ trang thiết bị. | REAL | ✓ | | | | |
| 8 ⚠️ | 2407 | **FAKE** BẪY | Cha của nam sinh viên trên ở Trung Quốc đã qua đời bốn năm trước đã hồi âm tin nhắn của con trai. | REAL | ✗ | | | | |
| 9 ⚠️ | 7539 | **REAL** BẪY | Ca sỹ Hoàng Thùy Linh được xướng tên liên tục ở các hạng mục: Nữ ca sỹ của năm, Bài hát của năm, Album của năm. | FAKE | ✗ | | | | |
| 10 | 2857 | REAL | Báo Nông thôn ngày nay đã có công văn hỏa tốc đề nghị các cơ quan chức năng tỉnh Hòa Bình nhanh chóng vào cuộc, xác minh làm rõ vụ việc nhóm phóng viên bị người của Nhà máy hành hung, cản trở làm việc. | UNCERTAIN | ✗ | | | | |
| 11 | 1723 | REAL | Ông Vũ Thanh Mai, Chánh Văn phòng Ban Tuyên giáo Trung ương được Ban Bí thư quyết định bổ nhiệm giữ chức Phó Trưởng ban Tuyên giáo Trung ương | REAL | ✓ | | | | |
| 12 | 2104 | REAL | Cục Thuế Thái Lan đang nghiên cứu các biện pháp để đánh thuế carbon bằng cách đưa ra các biện pháp và mức thuế rõ ràng đối với sản phẩm liên quan đến việc phát thải khí carbon (CO2) trong quá trình sản xuất. | REAL | ✓ | | | | |
| 13 ⚠️ | 1463 | **FAKE** BẪY | Trong khoảng 1.400 hộ dân bị ảnh hưởng bởi dự án, chỉ có 416/1.221 trường hợp được phê duyệt kiểm đếm, còn lại chưa được xem xét pháp lý vì những khu TĐC đang trong giai đoạn hoàn thiện, nền nhà chưa được áp giá nên không đủ điều kiện bàn giao cho hộ dân, ảnh hưởng rất lớn đến khâu bồi thường, hỗ trợ và TĐC. | REAL | ✗ | | | | |
| 14 | 7212 | REAL | Đại tá Đặng Hồng Đức cho biết những đổi mới này phù hợp với Nghị quyết số 12, ngày 16/3/2022 của Bộ Chính trị về đẩy mạnh xây dựng lực lượng công an nhân dân thật sự trong sạch, vững mạnh, chính quy. | REAL | ✓ | | | | |
| 15 | 1402 | REAL | Hiện nay, có khoảng 80% lao động trong ngành Du lịch đã qua đào tạo, bồi dưỡng chuyên môn, nghiệp vụ về du lịch. | REAL | ✓ | | | | |
| 16 | 2852 | FAKE | Thông tin từ Tổ chức động vật châu Á cho biết 2 voi này gồm voi đực Ta Nuôn nay đã 40 tuổi, trước đó thuộc sở hữu của ông Y Khu (xã Krông Na, H.Buôn Đôn). | FAKE | ✓ | | | | |
| 17 | 5760 | REAL | Thư Kỳ tham gia một sự kiện thời trang tại Trung Quốc và khoe vóc dáng cong vút với bộ đồ tinh tế. | REAL | ✓ | | | | |
| 18 | 3542 | FAKE | Tường Anh theo bố mẹ sinh sống, học tập tại nhiều nước như Anh, Pháp, Đức và học hỏi được nhiều điều từ bố mẹ là chuyên gia hàng đầu tại một tổ chức nghiên cứu quốc tế | UNCERTAIN | ✗ | | | | |
| 19 | 2647 | REAL | BCRA cùng các nhà thám hiểm đến từ Anh, Australia, New Zealand sau 3 tuần làm việc thám hiểm hang động ở khu vực Phong Nha Kẻ Bàng, huyện Tuyên Hoá và huyện Minh Hoá, Quảng Bình, đã phát hiện 22 hang động. | REAL | ✓ | | | | |
| 20 | 3799 | FAKE | N. sử dụng tài khoản Facebook cá nhân của mình với tên NNN để vu khống Chủ tịch UBND huyện Cái Nước, TAND huyện Cái Nước làm ăn bất chính, ức hiếp nhân dân. | UNCERTAIN | ✗ | | | | |

---

## Summary

| Metric | V8 (Baseline) | V10 TRUST Agent | Competitor_A | Competitor_B |
|---|---|---|---|---|
| **Accuracy** | **30.0%** (6/20) | **60.0%** (12/20) | | |
| REAL as REAL | 2/14 | 10/14 | | |
| FAKE as FAKE | 4/6 | 2/6 | | |
| UNCERTAIN rate | 14/20 (70%) | 4/20 (20%) | | |
| Avg confidence (correct) | — | — | | |
| Avg confidence (wrong) | — | — | | |

---

## Bẫy Categories Legend

| Symbol | Meaning | Description |
|---|---|---|
| ⚠️ | Bẫy cases | Cases where ground truth is counter-intuitive or evidence is sparse |
| ✗ | Wrong | V10 predicted different from expected label |
| ✓ | Correct | V10 predicted same as expected label |

### Bẫy Analysis

| Bẫy ID | Why it's tricky | What V10 did wrong |
|---|---|---|
| 1172 | Claim nói **66 trận, 40 bàn** nhưng số thực Mbappe khác | V10 không có đủ evidence số liệu → chủ đề + tên khớp → bias REAL |
| 2407 | "Cha đã qua đời **hồi âm tin nhắn**" — bất hợp lý logic nhưng evidence mơ hồ | Search trả về bài về scam Trung Quốc không rõ → bias REAL |
| 7539 | Hoàng Thùy Linh — evidence có thể bị chi phối bởi scandal cũ | Evidence trả về bài tiêu cực → bias FAKE |
| 1463 | Nhiều con số (1.400, 416, 1.221) — evidence không có số chính xác | Chủ đề khớp + số mơ hồ → bias REAL |

---