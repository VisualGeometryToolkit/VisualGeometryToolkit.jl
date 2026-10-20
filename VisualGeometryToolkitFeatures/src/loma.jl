import ONNXRunTime as ORT

struct LoMaBDescriptor
    max_detections::Int
    descriptor_model::ORT.InferenceSession

    LoMaBDescriptor() = new(2048, ORT.load_inference(download("https://github.com/davnords/storage/releases/download/loma/loma_descriptor_dedode_b.onnx"), execution_provider=:cuda))

    LoMaBDescriptor(max_detections) = new(max_detections, ORT.load_inference(download("https://github.com/davnords/storage/releases/download/loma/loma_descriptor_dedode_b.onnx"), execution_provider=:cuda))
end

function (l::LoMaBDescriptor)(img, detections)
    h, w, _... = size(img)
    _detections = Float32.(stack(map(det -> det.center ./ [h, w] .* 2 .- 1, detections[1:l.max_detections]), dims=1))
    _detections = insertdims(_detections, dims=1)
    img = insertdims(img, dims=1)

    descriptor_inputs = l.descriptor_model.input_names

    descriptors = l.descriptor_model(Dict(
        descriptor_inputs[1] => img,
        descriptor_inputs[2] => _detections
    ))["descriptions"]
    described_features(
        detections[1:l.max_detections],
        dropdims(descriptors, dims=1)
    )
end

struct LoMaBMatcher
    matcher_model::ORT.InferenceSession

    LoMaBMatcher() = new(ORT.load_inference(download("https://github.com/davnords/storage/releases/download/loma/loma_matcher_B128.onnx"), execution_provider = :cuda))
end

function (l::LoMaBMatcher)(features_a, features_b, size_a, size_b)
    matcher_inputs = l.matcher_model.input_names

    h_a, w_a, _... = size_a
    h_b, w_b, _... = size_b

    _detections_a = insertdims(Float32.(stack(map(feat -> feat.detection.center ./ [h_a, w_a] .* 2 .- 1, features_a), dims=1)), dims=1)
    _detections_b = insertdims(Float32.(stack(map(feat -> feat.detection.center ./ [h_b, w_b] .* 2 .- 1, features_b), dims=1)), dims=1)

    desc_a = insertdims(Float32.(stack(map(feat -> feat.descriptor, features_a), dims=1)), dims=1)
    desc_b =  insertdims(Float32.(stack(map(feat -> feat.descriptor, features_b), dims=1)), dims=1)

    l.matcher_model(Dict(
        matcher_inputs[1] => _detections_a,
        matcher_inputs[2] => _detections_b,
        matcher_inputs[3] => desc_a,
        matcher_inputs[4] => desc_b
    ))
end
