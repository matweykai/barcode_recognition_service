def test_barcode_analyzer_service(barcode_analyzer, barcode_data, crop_barcode_data):
    test_img, _ = barcode_data
    _, text = crop_barcode_data

    pred_list = barcode_analyzer.find_barcodes(test_img)

    assert len(pred_list) == 1
    assert sum([item[0] == item[1] for item in zip(pred_list[0].text, text)]) > 10
