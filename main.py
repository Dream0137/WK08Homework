# load sample dataset
from pycaret.datasets import get_data
data = get_data('diabetes')

from pycaret.classification import setup, compare_models, evaluate_model, plot_model, predict_model, save_model, load_model
s = setup(data, target = 'Class variable', session_id = 123)

from pycaret.classification import ClassificationExperiment
s = ClassificationExperiment()
s.setup(data, target = 'Class variable', session_id = 123)


# functional API
best = compare_models()

# OOP API
best = s.compare_models()


# functional API
evaluate_model(best)

# OOP API
s.evaluate_model(best)


# functional API
plot_model(best, plot = 'auc')

# OOP API
s.plot_model(best, plot = 'auc')

# functional API
plot_model(best, plot = 'confusion_matrix')

# OOP API
s.plot_model(best, plot = 'confusion_matrix')

# functional API
predict_model(best)

# OOP API
s.predict_model(best)


# functional API
predictions = predict_model(best, data=data)
predictions.head()

# OOP API
predictions = s.predict_model(best, data=data)
predictions.head()

# functional API
predictions = predict_model(best, data=data, raw_score=True)
predictions.head()

# OOP API
predictions = s.predict_model(best, data=data, raw_score=True)
predictions.head()


# functional API
save_model(best, 'my_best_pipeline')

# OOP API
s.save_model(best, 'my_best_pipeline')

# functional API
loaded_model = load_model('my_best_pipeline')
print(loaded_model)

# OOP API
loaded_model = s.load_model('my_best_pipeline')
print(loaded_model)