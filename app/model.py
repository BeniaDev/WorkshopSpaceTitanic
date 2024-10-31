import fire
import pandas as pd
from catboost import CatBoostClassifier
import os
from sklearn.impute import KNNImputer
def log(message):
    print(f"[LOG] {message}")

class SpaceTitanicModel:
    def __init__(self):
        self.default_model_path = 'models/catboost_model_best.cbm'
        self.models = CatBoostClassifier().load_model(self.default_model_path)

    def preprocess(self, test:pd.DataFrame) -> pd.DataFrame:
        train = pd.read_csv('./data/train.csv')
        train.drop('Transported',axis=1,inplace=True)
        WITH=pd.concat([train,test])
        WITH.drop(['PassengerId','Cabin','Name'],axis=1,inplace=True)
        WITH.replace({'VIP' : {False : 0, True : 1}},inplace=True)
        WITH.replace({'CryoSleep' : {False : 0, True : 1}},inplace=True)
        WITH.replace({'HomePlanet' : {'Europa' : 0, 'Earth' : 1,'Mars': 2}},inplace=True)
        WITH.replace({'Destination' : {'TRAPPIST-1e' : 0, 'PSO J318.5-22' : 1,'55 Cancri e': 2}},inplace=True)

        imputer = KNNImputer(n_neighbors=1, weights="uniform")

        l=imputer.fit_transform(WITH)

        WITH1=pd.DataFrame(l,columns=WITH.columns)
        ind=range(12970)
        WITH1['Index']=ind
        WITH1=WITH1.set_index('Index')

        Home_planet=pd.get_dummies(WITH1.HomePlanet).add_prefix('HomePlanet')
        WITH1=WITH1.merge(Home_planet,on='Index')
        WITH1=WITH1.drop(['HomePlanet'],axis=1)

        Destination=pd.get_dummies(WITH1.Destination).add_prefix('Destination')
        WITH1=WITH1.merge(Destination,on='Index')
        WITH1=WITH1.drop(['Destination'],axis=1)

        test1=WITH1[8693:]

        return test1
        
        

    def train(self, 
              data_path, 
              target_column, 
              model_path=None,
              iterations=1000,
              learning_rate=0.1,
              depth=6):
        """Train the model and save it
        
        Args:
            data_path (str): Path to training data (CSV)
            target_column (str): Name of the target column
            model_path (str, optional): Where to save the model
            iterations (int, optional): Number of trees
            learning_rate (float, optional): Learning rate
            depth (int, optional): Depth of trees
        """

        return self.models

    def predict(self, 
                data_path, 
                model_path=None,
                output_path='predictions.csv'):
        """Make predictions on new data
        
        Args:
            data_path (str): Path to data for prediction (CSV)
            model_path (str, optional): Path to saved model
            output_path (str, optional): Where to save predictions
        """
        model_path = model_path or self.default_model_path
        
        try:
            # Load and prepare data
            df = pd.read_csv(data_path)
            log(df.head(5))

            df = self.preprocess(df)
            log(df.head(5))
            # Make predictions
            predictions = self.models.predict(df)

            # Save predictions
            pd.DataFrame(predictions, columns=['prediction']).to_csv(output_path, index=False)
            log(f"Predictions saved to {output_path}")
            
        except FileNotFoundError:
            log(f"Error: Could not find file {data_path} or {model_path}")
        except Exception as e:
            log(f"Error during prediction: {str(e)}")