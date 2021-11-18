# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import onnx
import logging
import argparse
from pathlib import Path
from onnx import helper
import numpy as np
from transformers import GPT2Config
from gpt2_helper import PRETRAINED_GPT2_MODELS
from convert_to_onnx import main as convert_gpt2_to_onnx
from benchmark_helper import Precision
"""
This converts GPT2 model to onnx with beam search operator.

Examples:
   python convert_beam_search.py -m gpt2 --gpt2_onnx .\onnx_models\gpt2_past_fp32.onnx --output .\onnx_models\gpt2_beam_search.onnx --output_sequences_scores
"""

config: GPT2Config = None

logger = logging.getLogger('')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name_or_path',
                        required=True,
                        type=str,
                        help='Model path, or pretrained model name in the list: ' + ', '.join(PRETRAINED_GPT2_MODELS))

    parser.add_argument('--cache_dir',
                        required=False,
                        type=str,
                        default=os.path.join('.', 'cache_models'),
                        help='Directory to cache pre-trained models')

    parser.add_argument('--gpt2_onnx',
                        required=True,
                        type=str,
                        help='Output directory for GPT-2 onnx model, or model path ends with .onnx')

    parser.add_argument('--output',
                        required=False,
                        type=str,
                        help='Output directory for beam search model, or model path ends with .onnx')

    parser.add_argument("-p",
                        "--precision",
                        required=False,
                        type=Precision,
                        default=Precision.FLOAT32,
                        choices=[Precision.FLOAT32, Precision.FLOAT16],
                        help="Precision of model to run. fp32 for full precision, fp16 for half or mixed precision")

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('-e', '--use_external_data_format', required=False, action='store_true')
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument('--run_baseline', required=False, action='store_true', help="run huggingface beam search")
    parser.set_defaults(run_baseline=False)

    beam_search_group = parser.add_argument_group("beam search options")

    beam_search_group.add_argument('--output_sequences_scores',
                                   required=False,
                                   action='store_true',
                                   help="output sequences scores")
    beam_search_group.set_defaults(output_sequences_scores=False)

    beam_search_group.add_argument('--output_token_scores',
                                   required=False,
                                   action='store_true',
                                   help="output token scores")
    beam_search_group.set_defaults(output_token_scores=False)

    beam_search_group.add_argument('--early_stopping', required=False, action='store_true')
    beam_search_group.set_defaults(early_stopping=False)

    beam_search_group.add_argument('--min_length', type=int, required=False, default=1, help='Min sequence length')

    beam_search_group.add_argument('--max_length', type=int, required=False, default=50, help='Max sequence length')

    beam_search_group.add_argument('--no_repeat_ngram_size',
                                   type=int,
                                   required=False,
                                   default=0,
                                   help='No repeat ngram size')

    beam_search_group.add_argument('--num_beams', type=int, required=False, default=4, help='Beam size')

    beam_search_group.add_argument('--num_return_sequences',
                                   type=int,
                                   required=False,
                                   default=1,
                                   help='Number of return sequence')

    beam_search_group.add_argument('--temperature',
                                   type=float,
                                   required=False,
                                   default=1,
                                   help='Softmax temperature for output logits.')

    beam_search_group.add_argument('--length_penalty',
                                   type=float,
                                   required=False,
                                   default=1,
                                   help='Positive. >1 to penalize and <1 to encorage short sentence.')

    beam_search_group.add_argument('--repetition_penalty',
                                   type=float,
                                   required=False,
                                   default=1,
                                   help='Positive. >1 to penalize and <1 to encorage.')

    mixed_precision_option_grapu = parser.add_argument_group(
        "mixed precision conversion parameters that works when \"--precision fp16\" is specified")

    mixed_precision_option_grapu.add_argument('--io_block_list',
                                              nargs='+',
                                              required=False,
                                              default=[],
                                              help='List of inputs or outputs in float32')

    mixed_precision_option_grapu.add_argument(
        '--op_block_list',
        nargs='+',
        required=False,
        default=[],
        help='List of operators (like Add LayerNormalization FastGelu) to compute in float32.')

    mixed_precision_option_grapu.add_argument('--node_block_list',
                                              nargs='+',
                                              required=False,
                                              default=[],
                                              help='List of node names to compute in float32.')

    mixed_precision_option_grapu.add_argument('--force_fp16_initializers',
                                              required=False,
                                              action='store_true',
                                              help='Convert all float initializers to float16.')
    mixed_precision_option_grapu.set_defaults(force_fp16_initializers=False)

    args = parser.parse_args(argv)

    return args


def gpt2_to_onnx(args):
    model_name = args.model_name_or_path

    print(f"use convert_to_onnx.py to convert model {model_name} to onnx {args.gpt2_onnx} ...")
    arguments = [
        '--model_name_or_path', model_name, '--output', args.gpt2_onnx, '--optimize_onnx', '--precision',
        'fp32' if args.precision == Precision.FLOAT32 else 'fp16', '--test_runs', '1', '--test_cases', '10'
    ]
    if args.use_gpu:
        arguments.append('--use_gpu')
    if args.use_external_data_format:
        arguments.append('--use_external_data_format')

    # mixed precision conversion options
    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 or mixed precision model cannot run in CPU. Please add --use_gpu"
        if args.io_block_list:
            arguments.append('--io_block_list')
            arguments.extend(args.io_block_list)
        if args.op_block_list:
            arguments.append('--op_block_list')
            arguments.extend(args.op_block_list)
        if args.node_block_list:
            arguments.append('--node_block_list')
            arguments.extend(args.node_block_list)
        if args.force_fp16_initializers:
            arguments.append('--force_fp16_initializers')

    convert_gpt2_to_onnx(arguments)

    # Run symbolic shape inference to walk around ORT shape inference issue for subgraph.
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    out = SymbolicShapeInference.infer_shapes(onnx.load(args.gpt2_onnx), auto_merge=True, guess_output_rank=False)
    if out:
        onnx.save(out, args.gpt2_onnx)


def create_ort_session(model_path, use_gpu):
    from onnxruntime import SessionOptions, InferenceSession, __version__ as ort_version, GraphOptimizationLevel
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

    ort_session = InferenceSession(model_path, sess_options, providers=execution_providers)
    return ort_session


def convert_model(args):
    if os.path.exists(args.gpt2_onnx):
        print(f"skip convert_to_onnx since path existed: {args.gpt2_onnx}")
    else:
        gpt2_to_onnx(args)

    #create_ort_session(args.gpt2_onnx, args.use_gpu)

    global config
    config = GPT2Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    print(config)

    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id
    vocab_size = config.vocab_size

    model = onnx.load(args.gpt2_onnx)
    model.graph.name = "gpt2 subgraph"
    inputs = [
        "input_ids", "max_length", "min_length", "num_beams", "num_return_sequences", "temperature", "length_penalty",
        "repetition_penalty", "vocab_mask"
    ]

    outputs = ["sequences"]
    if args.output_sequences_scores:
        outputs.append("sequences_scores")
    if args.output_token_scores:
        outputs.append("scores")

    node = helper.make_node('BeamSearch', inputs=inputs, outputs=outputs, name='BeamSearch_GPT2')
    node.domain = "com.microsoft"
    node.attribute.extend([
        helper.make_attribute("eos_token_id", eos_token_id),
        helper.make_attribute("pad_token_id", pad_token_id),
        helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
        helper.make_attribute("early_stopping", 1 if args.early_stopping else 0),
        helper.make_attribute("body", model.graph),
    ])

    from onnx import TensorProto

    # graph inputs
    input_ids = helper.make_tensor_value_info('input_ids', TensorProto.INT32, ['batch_size', 'sequence_length'])
    max_length = helper.make_tensor_value_info('max_length', TensorProto.INT32, [1])
    min_length = helper.make_tensor_value_info('min_length', TensorProto.INT32, [1])
    num_beams = helper.make_tensor_value_info('num_beams', TensorProto.INT32, [1])
    num_return_sequences = helper.make_tensor_value_info('num_return_sequences', TensorProto.INT32, [1])
    temperature = helper.make_tensor_value_info('temperature', TensorProto.FLOAT, [1])
    length_penalty = helper.make_tensor_value_info('length_penalty', TensorProto.FLOAT, [1])
    repetition_penalty = helper.make_tensor_value_info('repetition_penalty', TensorProto.FLOAT, [1])
    vocab_mask = helper.make_tensor_value_info('vocab_mask', TensorProto.INT32, [vocab_size])

    graph_inputs = [
        input_ids, max_length, min_length, num_beams, num_return_sequences, temperature, length_penalty,
        repetition_penalty, vocab_mask
    ]

    # graph outputs
    sequences = helper.make_tensor_value_info('sequences', TensorProto.INT32,
                                              ['batch_size', 'num_return_sequences', 'max_length'])

    sequences_scores = helper.make_tensor_value_info('sequences_scores', TensorProto.FLOAT,
                                                     ['batch_size', 'num_return_sequences'])
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT,
                                           ['max_length - sequence_length', 'batch_size', 'num_beams', vocab_size])

    initializers = []

    graph_outputs = [sequences]

    if args.output_sequences_scores:
        graph_outputs.append(sequences_scores)

    if args.output_token_scores:
        graph_outputs.append(scores)

    new_graph = helper.make_graph([node], 'gpt2-beam-search', graph_inputs, graph_outputs, initializers)

    # Create the model
    new_model = helper.make_model(new_graph, producer_name='onnxruntime.transformers', opset_imports=model.opset_import)
    onnx.save(new_model, args.output)


def test_model(args):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path,
                                            cache_dir=args.cache_dir,
                                            pad_token_id=tokenizer.eos_token_id)
    input_ids = tokenizer.encode('I enjoy walking in the park', return_tensors='pt')

    global config
    if config is None:
        config = GPT2Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id
    vocab_size = config.vocab_size

    if args.run_baseline:
        print('-' * 50)
        print("Test PyTorch model and beam search with huggingface transformers...")
        beam_outputs = model.generate(input_ids,
                                      max_length=args.max_length,
                                      min_length=args.min_length,
                                      num_beams=args.num_beams,
                                      early_stopping=args.early_stopping,
                                      no_repeat_ngram_size=args.no_repeat_ngram_size,
                                      eos_token_id=eos_token_id,
                                      pad_token_id=pad_token_id,
                                      num_return_sequences=args.num_return_sequences,
                                      temperature=args.temperature,
                                      length_penalty=args.length_penalty,
                                      repetition_penalty=args.repetition_penalty)
        print("input_ids", input_ids)
        print("huggingface transformers output:", beam_outputs)
        for i, beam_output in enumerate(beam_outputs):
            print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

    print('-' * 50)
    print("Test ONNX model and bream search with onnxruntime...")

    # TODO: remove debug code
    import time
    print('You have 15 seconds to attach a debugger.')
    time.sleep(15)

    ort_session = create_ort_session(args.output, args.use_gpu)

    batch_size = 1
    input_ids = input_ids.repeat(batch_size, 1)

    inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int32),
        "max_length": np.array([args.max_length], dtype=np.int32),
        "min_length": np.array([args.min_length], dtype=np.int32),
        "num_beams": np.array([args.num_beams], dtype=np.int32),
        "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
        "temperature": np.array([args.temperature], dtype=np.float32),
        "length_penalty": np.array([args.length_penalty], dtype=np.float32),
        "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
        "vocab_mask": np.ones((vocab_size), dtype=np.int32)
    }

    test_data_dir = Path(args.output).parent.as_posix()
    print("test_data_dir", test_data_dir)
    from bert_test_data import output_test_data
    all_inputs = [inputs]
    for i, inputs in enumerate(all_inputs):
        dir = os.path.join(test_data_dir, 'test_data_set_' + str(i))
        output_test_data(dir, inputs)

    print("inputs", inputs)
    result = ort_session.run(None, inputs)

    sequences = result[0]
    print("outputs", sequences)

    #TODO: print all sequences. Below shows only the first one
    first_sequence = tokenizer.decode(sequences[0][0], skip_special_tokens=True)
    print(first_sequence)


def main():
    args = parse_arguments()

    if os.path.exists(args.output):
        print(f"skip conversion since path existed: {args.output}")
    else:
        convert_model(args)

    test_model(args)


if __name__ == '__main__':
    main()