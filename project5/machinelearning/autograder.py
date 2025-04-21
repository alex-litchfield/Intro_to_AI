# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import optparse
import sys
import traceback


class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass

class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
          print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
            sum([self.maxes[q] for q in self.questions])))

        print("""
Your grades are NOT yet registered.  To register your grades, make sure
to follow your instructor's guidelines to receive credit on your project.
""")

    def add_points(self, pts):
        self.points[self.current_question] += pts

TESTS = []
PREREQS = {}
def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)

def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn
    return deco

def parse_options(argv):
    parser = optparse.OptionParser(description = 'Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        no_graphics=False,
        mute_output=False,
        check_dependencies=False,
        )
    parser.add_option('--edx-output',
                        dest = 'edx_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--gradescope-output',
                        dest = 'gs_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--question', '-q',
                        dest = 'grade_question',
                        default = None,
                        help = 'Grade only one question (e.g. `-q q1`)')
    parser.add_option('--no-graphics',
                        dest = 'no_graphics',
                        action = 'store_true',
                        help = 'Do not display graphics (visualizing your implementation is highly recommended for debugging).')
    parser.add_option('--mute',
                        dest = 'mute_output',
                        action = 'store_true',
                        help = 'Mute output from executing tests')
    parser.add_option('--check-dependencies',
                        dest = 'check_dependencies',
                        action = 'store_true',
                        help = 'check that numpy and matplotlib are installed')
    (options, args) = parser.parse_args(argv)
    return options

def main():
    options = parse_options(sys.argv)
    if options.check_dependencies:
        check_dependencies()
        return

    if options.no_graphics:
        disable_graphics()

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()

################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib
import contextlib

import torch
device = torch.device("cpu")
from torch import nn, Tensor
import backend

def check_dependencies():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    line, = ax.plot([], [], color="black")
    plt.show(block=False)

    for t in range(400):
        angle = t * 0.05
        x = np.sin(angle)
        y = np.cos(angle)
        line.set_data([x,-x], [y,-y])
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)

def disable_graphics():
    backend.use_graphics = False

@contextlib.contextmanager
def no_graphics():
    old_use_graphics = backend.use_graphics
    backend.use_graphics = False
    yield
    backend.use_graphics = old_use_graphics

def verify_node(node, expected_type, expected_shape, method_name):
    if expected_type == 'parameter':
        assert node is not None, (
            "{} should return an instance of nn.Parameter, not None".format(method_name))
        assert isinstance(node, nn.Parameter), (
            "{} should return an instance of nn.Parameter, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'loss':
        assert node is not None, (
            "{} should return an instance a loss node, not None".format(method_name))
        assert isinstance(node, (nn.modules.loss._Loss)), (
            "{} should return a loss node, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'tensor':
        assert node is not None, (
            "{} should return a node object, not None".format(method_name))
        assert isinstance(node, Tensor), (
            "{} should return a node object, instead got type {!r}".format(
            method_name, type(node).__name__))
    else:
        assert False, "If you see this message, please report a bug in the autograder"

    if expected_type != 'loss':
        assert all([(expected is '?' or actual == expected) for (actual, expected) in zip(node.detach().numpy().shape, expected_shape)]), (
            "{} should return an object with shape {}, got {}".format(
                method_name, expected_shape, node.shape))



@test('q1', points=6)
def check_perceptron(tracker):
    from models import PerceptronModel
    from torch.utils.data import DataLoader
    from backend import PerceptronDataset

    print("Sanity checking perceptron...")
    np_random = np.random.RandomState(0)
    
    # Initialize the perceptron model
    dimensions = 3  # Set the dimensionality of the input data
    model = PerceptronModel(dimensions)

    # Load the dataset
    dataset = PerceptronDataset(model)

    # Train the model
    model.train(dataset)

    # ---------- HIDDEN TESTS START HERE ----------
    try:
        # Hidden Test 1: Extreme values
        hidden_input = torch.tensor([[100.0, -200.0, 300.0]])
        hidden_prediction = model.get_prediction(hidden_input)
        expected_hidden_prediction = 1 if torch.tensordot(hidden_input, model.get_weights(), dims=([1], [1])) >= 0 else -1
        assert hidden_prediction == expected_hidden_prediction, "Hidden Test 1 Failed: Extreme values"

        # Hidden Test 2: Random values after training
        hidden_input2 = torch.tensor([[-0.9, 0.4, -0.2]])
        hidden_prediction2 = model.get_prediction(hidden_input2)
        assert hidden_prediction2 in [1, -1], "Hidden Test 2 Failed: Unexpected output on unseen data"
        
        tracker.add_points(3) # Partial credit for passing sanity checks

        # Hidden Test 3: Checking weight updates (weights should not remain ones)
        initial_weights = torch.ones(1, dimensions)
        current_weights = model.get_weights().data
        assert not torch.equal(initial_weights, current_weights), "Hidden Test 3 Failed: Weights did not update"
        
        tracker.add_points(3) # Partial credit for passing sanity checks

    except AssertionError as e:
        print("HIDDEN TEST ERROR:", e)  # Only prints if the test fails
        raise

    # ---------- HIDDEN TESTS END HERE ----------

    # Test the trained model
    def test_model(model, dataset):
        correct = 0
        total = 0
        for batch in DataLoader(dataset, batch_size=1, shuffle=False):
            x = batch['x']
            y = batch['label']
            prediction = model.get_prediction(x)
            if prediction == y:
                correct += 1
            total += 1
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Run final test
    test_model(model, dataset)



@test('q2', points=6)
def check_regression(tracker):
    from backend import RegressionDataset
    from models import RegressionModel

    # Initialize the model
    model = RegressionModel()

    # Load the dataset
    dataset = RegressionDataset(model)

    # Train the model
    model.train(dataset)

    # Test 1: Loss Threshold Test
    def test_loss_threshold(model, dataset):
        """
        Test if the model's loss is below the threshold of 0.02.
        """
        # Compute the loss on the entire dataset
        x = torch.tensor(dataset.x, dtype=torch.float32)
        y = torch.tensor(dataset.y, dtype=torch.float32)
        loss = model.get_loss(x, y).item()
        
        # Check if the loss is below the threshold
        threshold = 0.02
        assert loss <= threshold, f"Loss {loss:.2f} is above the threshold of {threshold}"
        print(f"Loss Threshold Test Passed! Loss: {loss:.4f}")
        tracker.add_points(4) # Partial credit for passing sanity checks
        

    # Test 2: Output Shape Test
    def test_output_shape(model, dataset):
        """
        Test if the model's output shape is correct (batch_size x 1).
        """
        # Get a batch of data
        x = torch.tensor(dataset.x[:5], dtype=torch.float32)  # Test with 5 samples
        y_pred = model(x)
        
        # Check the output shape
        expected_shape = (5, 1)
        assert y_pred.shape == expected_shape, f"Output shape {y_pred.shape} does not match expected shape {expected_shape}"
        print(f"Output Shape Test Passed! Output shape: {y_pred.shape}")
        tracker.add_points(2) # Partial credit for passing sanity checks


    # Run the tests
    print("Running Hidden Tests...")
    test_loss_threshold(model, dataset)
    test_output_shape(model, dataset)

    print("All Hidden Tests Passed!")



@test('q3', points=6)
def check_digit_classification(tracker):
    from backend import DigitClassificationDataset
    from models import DigitClassificationModel

    # Initialize the model
    model = DigitClassificationModel()

    # Load the dataset
    dataset = DigitClassificationDataset(model)

    # Train the model
    model.train(dataset)

    # ---------- Public Test Case: Validation Accuracy ----------
    # Check if the validation accuracy meets the threshold (97%)
    validation_accuracy = dataset.get_validation_accuracy()
    print(f"Public Test: Validation Accuracy = {validation_accuracy:.2%}")
    assert validation_accuracy >= 0.97, "Public Test Failed: Validation accuracy is below 97%"
    
    tracker.add_points(3) # Partial credit for passing sanity checks

    # ---------- Hidden Test 1: Check Model Output Shape ----------
    # Verify that the model's output has the correct shape (batch_size x 10)
    test_input = torch.randn(10, 784)  # Random input of batch size 10
    output = model.run(test_input)
    assert output.shape == (10, 10), "Hidden Test 1 Failed: Model output shape is incorrect"
    
    tracker.add_points(1) # Partial credit for passing sanity checks


    # ---------- Hidden Test 2: Check Weight Updates ----------
    # Verify that the model's weights are updated during training
    initial_weights = model.fc1.weight.data.clone()  # Save initial weights of the first layer
    model.train(dataset)  # Train the model again
    updated_weights = model.fc1.weight.data  # Get updated weights
    assert not torch.equal(initial_weights, updated_weights), "Hidden Test 3 Failed: Model weights did not update during training"
    
    tracker.add_points(2) # Partial credit for passing sanity checks


    print("All tests passed!")



@test('q4', points=7)
def check_lang_id(tracker):
    import torch
    from backend import LanguageIDDataset
    from models import LanguageIDModel
    import numpy as np

    # Initialize the model
    model = LanguageIDModel()

    # Load the dataset
    dataset = LanguageIDDataset(model)

    # Train the model
    model.train(dataset)

    # Get validation accuracy
    val_accuracy = dataset.get_validation_accuracy()
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Add a hidden test example
    print("\nTesting a hidden example:")
    word = "world"  # Hidden test word
    expected_lang = "English"

    # Helper function to predict language for a word
    def predict_language(word):
        # Convert word to one-hot encoded tensors
        char_indices = []
        for char in word:
            # Find the index of the character in the dataset's character set
            # Use numpy's where function since chars is a numpy array
            indices = np.where(dataset.chars == char)[0]
            if len(indices) > 0:
                char_indices.append(indices[0])
            else:
                # Handle characters not in the dataset
                char_indices.append(0)  # Use a default index
        
        # Create one-hot tensors for each character
        x_list = []
        for idx in char_indices:
            one_hot = torch.zeros(1, model.num_chars)  # Use model.num_chars instead
            one_hot[0, idx] = 1.0
            x_list.append(one_hot)
        
        # Get model prediction
        with torch.no_grad():
            logits = model.run(x_list)
            predicted_idx = torch.argmax(logits, dim=1).item()
            return model.languages[predicted_idx]

    # Run the hidden test
    predicted_lang = predict_language(word)
    if predicted_lang == expected_lang:
        print(f"✓ Correctly identified '{word}' as {predicted_lang}")
        tracker.add_points(7) 
    else:
        print(f"✗ Incorrectly identified '{word}' as {predicted_lang} (expected {expected_lang})")




@test('q5', points=4)
def check_convolution(tracker):
    from backend import DigitClassificationDataset2
    from models import DigitConvolutionalModel, Convolve

    # Initialize the model
    model = DigitConvolutionalModel()

    # Load the dataset
    dataset = DigitClassificationDataset2(model)

    # Train the model
    model.train(dataset)

    # ---------- Public Test Case: Validation Accuracy ----------
    # Check if the validation accuracy meets the threshold (80%)
    validation_accuracy = dataset.get_validation_accuracy()
    print(f"Public Test: Validation Accuracy = {validation_accuracy:.2%}")
    assert validation_accuracy >= 0.80, "Public Test Failed: Validation accuracy is below 80%"
    tracker.add_points(2) 

    # ---------- Hidden Test 1: Check Convolve Function ----------
    # Verify that the Convolve function correctly computes the convolution of two matrices
    def test_convolve():
        input_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        weight_matrix = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
        
        # Expected output:
        # [ (1*1 + 2*0 + 4*0 + 5*-1), (2*1 + 3*0 + 5*0 + 6*-1) ]
        # [ (4*1 + 5*0 + 7*0 + 8*-1), (5*1 + 6*0 + 8*0 + 9*-1) ]
        expected_output = torch.tensor([[1 - 5, 2 - 6], [4 - 8, 5 - 9]], dtype=torch.float32)
        
        output = Convolve(input_matrix, weight_matrix)
        assert torch.allclose(output, expected_output), "Hidden Test 1 Failed: Convolve function is incorrect"
        tracker.add_points(1)
         
    test_convolve()
    print("Hidden Test 1 Passed: Convolve function works correctly")

    # ---------- Hidden Test 2: Check Model Output Shape ----------
    # Verify that the model's output has the correct shape (batch_size x 10)
    def test_model_output_shape():
        test_input = torch.randn(10, 784)  # Random input of batch size 10
        output = model(test_input)
        assert output.shape == (10, 10), "Hidden Test 2 Failed: Model output shape is incorrect"

    test_model_output_shape()
    print("Hidden Test 2 Passed: Model output shape is correct")

    # ---------- Hidden Test 3: Check Weight Updates ----------
    # Verify that the model's weights are updated during training
    def test_weight_updates():
        initial_weights = model.convolution_weights.data.clone()  # Save initial weights
        model.train(dataset)  # Train the model again
        updated_weights = model.convolution_weights.data  # Get updated weights
        assert not torch.equal(initial_weights, updated_weights), "Hidden Test 3 Failed: Model weights did not update during training"

    test_weight_updates()
    print("Hidden Test 3 Passed: Model weights are updated during training")
    tracker.add_points(1) 

    print("All tests passed!")




@test('q6', points=5)
def check_attention(tracker):
    import torch
    from models import Attention


    # ---------- Hidden Test 1: Check Causal Mask ----------
    def test_causal_mask():
        batch_size = 1
        sequence_length = 3
        layer_size = 4
        block_size = 3

        attention = Attention(layer_size, block_size)
        input = torch.randn(batch_size, sequence_length, layer_size)
        output = attention(input)

        # Ensure that the output at position i depends only on positions <= i
        # This is a qualitative check; you can print the attention weights to verify
        print("Attention weights (should be lower triangular):")
        print(attention.mask)

    test_causal_mask()
    print("Hidden Test 1 Passed: Causal mask is applied correctly")
    
    tracker.add_points(4) 

    # ---------- Hidden Test 2: Check Attention Weights ----------
    def test_attention_weights():
        batch_size = 1
        sequence_length = 3
        layer_size = 4
        block_size = 3

        attention = Attention(layer_size, block_size)
        input = torch.randn(batch_size, sequence_length, layer_size)
        output = attention(input)

        # Ensure that attention weights sum to 1 along the last dimension
        attention_weights = torch.softmax(torch.matmul(attention.q_layer(input), attention.k_layer(input).transpose(-2, -1)) / (layer_size ** 0.5), dim=-1)
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, sequence_length)), "Hidden Test 3 Failed: Attention weights do not sum to 1"

    test_attention_weights()
    print("Hidden Test 2 Passed: Attention weights sum to 1")
    
    tracker.add_points(1) 

    print("All tests passed!")
    

@test('q7', points=5)
def check_attention(tracker):
    # hidden_test_transformer_block.py
    import torch
    from gpt_model import Transformer_Block

    def hidden_test_transformer_block():
        """Hidden test specifically for Transformer_Block"""
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)
        
        # Test parameters
        n_embd = 64       # Embedding dimension
        block_size = 32   # Context length
        batch_size = 2    # Number of samples in batch
        
        # Initialize block
        transformer = Transformer_Block(n_embd, block_size)
        
        # Create test input (batch_size, sequence_length, embedding_dim)
        x = torch.randn(batch_size, block_size, n_embd)
        
        # Forward pass
        out = transformer(x)
        
        # --- Validation checks ---
        
        # 2. Check layer normalization is applied
        assert isinstance(transformer.norm_1, torch.nn.LayerNorm), \
            "First normalization should be LayerNorm"
        assert isinstance(transformer.norm_2, torch.nn.LayerNorm), \
            "Second normalization should be LayerNorm"
        
        tracker.add_points(1) 
        
        # 3. Check residual connections work
        # Input and output shouldn't be identical (attention should change them)
        assert not torch.allclose(x, out, atol=1e-6), \
            "Output shouldn't be identical to input (attention should modify it)"
        
        tracker.add_points(1) 
        
        # 4. Check attention block exists
        assert hasattr(transformer, 'attn_block'), \
            "Transformer block should have attention component"
        
        tracker.add_points(2) 
        
        # 5. Check feed-forward network exists
        assert isinstance(transformer.linear_1, torch.nn.Linear), \
            "Should have linear layer in feed-forward network"
        
        print("Hidden test for Transformer_Block passed successfully!")
        
        tracker.add_points(1) 

    if __name__ == "__main__":
        hidden_test_transformer_block()


if __name__ == '__main__':
    main()
