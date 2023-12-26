def test_real_img_inference(barcode_recognizer, crop_barcode_data):
    crop_barcode_img, barcode_text = crop_barcode_data

    pred = barcode_recognizer.decode(crop_barcode_img)

    assert sum([item[0] == item[1] for item in zip(pred, barcode_text)]) > 10
