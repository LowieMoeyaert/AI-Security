import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.utils import to_categorical
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, CarliniL2Method, ProjectedGradientDescent
from tensorflow.keras.datasets import cifar10

# Disable TensorFlow eager execution (necessary for ART with TensorFlow 2.x)
tf.compat.v1.disable_eager_execution()

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images to [0,1]
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Select a pre-trained model (ResNet50 or VGG16)
def get_model(model_name="ResNet50"):
    if model_name == "ResNet50":
        model = ResNet50(weights="imagenet", include_top=True)  # include_top=True for full model
        preprocess_input = preprocess_resnet
    elif model_name == "VGG16":
        model = VGG16(weights="imagenet", include_top=True)
        preprocess_input = preprocess_vgg
    else:
        raise ValueError("Unsupported model. Choose 'ResNet50' or 'VGG16'.")
    
    # Ensure the model has a valid loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model, preprocess_input

# Load model and prepare ART classifier
model_name = "ResNet50"  # Change to "VGG16" if desired
model, preprocess_input = get_model(model_name)

# Pass preprocessing as None or a tuple (for example, (0.5, 0.5) for normalization)
art_classifier = KerasClassifier(model=model, preprocessing=preprocess_input)

# Adversarial attack examples
def generate_adversarial_examples(attack_method, x, y):
    attack = attack_method(classifier=art_classifier)
    x_adv = attack.generate(x=x)
    return x_adv

# Evaluate the impact of adversarial attacks
def evaluate_attack(attack_name, attack_method, x_sample, y_sample):
    print(f"\nEvaluating {attack_name} attack...")
    x_adv = generate_adversarial_examples(attack_method, x_sample, y_sample)
    predictions = np.argmax(art_classifier.predict(x_adv), axis=1)
    true_labels = np.argmax(y_sample, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Accuracy after {attack_name} attack: {accuracy:.2%}")
    return x_adv

# Subset of test samples
x_sample, y_sample = x_test[:100], y_test[:100]
x_sample_preprocessed = preprocess_input(x_sample)

# FGSM Attack
fgsm_attack = lambda classifier: FastGradientMethod(classifier, eps=0.1)
x_adv_fgsm = evaluate_attack("FGSM", fgsm_attack, x_sample_preprocessed, y_sample)

# PGD Attack
pgd_attack = lambda classifier: ProjectedGradientDescent(classifier, eps=0.1, max_iter=20)
x_adv_pgd = evaluate_attack("PGD", pgd_attack, x_sample_preprocessed, y_sample)

# Carlini & Wagner (C&W) Attack
cw_attack = lambda classifier: CarliniL2Method(classifier, confidence=0.1)
x_adv_cw = evaluate_attack("C&W", cw_attack, x_sample_preprocessed, y_sample)

# Visualize original and adversarial examples
def plot_examples(original, adversarial, labels, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original[0])
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(adversarial[0])
    axes[1].set_title(f"{title} Attack")
    axes[1].axis("off")
    plt.show()

plot_examples(x_sample, x_adv_fgsm, y_sample, "FGSM")
plot_examples(x_sample, x_adv_pgd, y_sample, "PGD")
plot_examples(x_sample, x_adv_cw, y_sample, "C&W")
