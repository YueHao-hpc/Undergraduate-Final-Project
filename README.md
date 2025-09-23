

Usage

1. Environment Setup

Install required dependencies:

pip install torch numpy scipy scikit-learn pandas matplotlib

2. Data Preparation
	•	Input data: current signals from railway switch machines (200 samples per cycle).
	•	Use excel_column_padding.py to preprocess raw Excel data.
	•	Use extract_features_time_freq.py to generate 16-dimensional time-frequency feature vectors.

3. Model Training
	•	MLP model (feature-based input):

python train_mlp_model.py


	•	BiGRU model (sequence-based input):

python train_bigru_model.py


	•	CNN–LSTM–Attention model (hybrid input):

python train_cnn_lstm_attention.py



Training scripts output learning curves, confusion matrices, and classification metrics.

⸻

Experimental Results

All three models achieved 98%+ accuracy on the validation dataset:
	•	MLP: Fast convergence, strong generalization, suitable for lightweight deployment.
	•	BiGRU: Captures sequential dependencies, stable on fault transitions.
	•	CNN–LSTM–Attention: Best performance, near-perfect classification, enhanced interpretability with attention mechanism.

⸻

Future Work
	•	Explore online learning and domain adaptation for cross-device and cross-environment robustness.
	•	Apply model compression (pruning, quantization, distillation) for efficient edge deployment.
	•	Incorporate explainable AI methods (e.g., saliency maps, attention visualization) to improve transparency and trust.


