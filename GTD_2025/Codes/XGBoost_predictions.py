import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer





class XGBoost_predictions:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path, encoding='ISO-8859-1')
        # sort by attack date and group
        self.data.sort_values(by=['enc_group', 'attack_date'], inplace=True)
        self.data.drop(columns=['attack_date'])
        #self.target = self.data['enc_group']
        #self.features = self.data.drop(columns=['enc_group'])
        self.clean_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.handle_leakage()

    def clean_data(self):
        """Remove non-numeric columns and impute missing values with the median."""
        #self.features = self.features.select_dtypes(include=['number'])
        #self.features = self.features.fillna(self.features.median())
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        columns_to_impute = numeric_columns.drop('enc_group', errors='ignore')  # 'errors=ignore' handles cases where 'enc_group' is not in the numeric columns

        # Create an imputer object to impute with the median
        imputer = SimpleImputer(strategy='median')
        non_numeric_columns = self.data.select_dtypes(exclude=['number']).columns.drop('enc_group', errors='ignore')
        self.data.drop(columns=non_numeric_columns, inplace=True)

        # Apply imputation on the selected numeric columns
        self.data[columns_to_impute] = imputer.fit_transform(self.data[columns_to_impute])


    def encode_target(self):
        """Convert categorical target values into numeric values using factorization."""
        if self.data['enc_group'].dtype == 'object':  # Ensures encoding only happens for categorical targets
            self.data['enc_group'], _ = pd.factorize(self.data['enc_group'])

    def splitting(self, train_df, test_df):
        #X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42, shuffle=True)
        y_train = train_df['enc_group']
        X_train = train_df.drop(columns=['enc_group'])
        y_test = test_df['enc_group']
        X_test = test_df.drop(columns=['enc_group'])
        return X_train, X_test, y_train, y_test
    
    def handle_leakage(self):
        train_frames = []
        test_frames = []
        for _, group_data in self.data.groupby('enc_group'):
            split_point = int(len(group_data) * 0.8)  # 80% for training
            train_frames.append(group_data.iloc[:split_point])
            test_frames.append(group_data.iloc[split_point:])           


        # Concatenate all the group-specific splits into final train and test DataFrames
        train_df = pd.concat(train_frames)
        test_df = pd.concat(test_frames)

        # Shuffle each DataFrame separately
        train_df = shuffle(train_df)
        test_df = shuffle(test_df)

        X_train, X_test, y_train, y_test = self.splitting(train_df, test_df)

        return X_train, X_test, y_train, y_test
    
    def randomizedSearch(self):
        n_estimators = [5, 10, 20, 50, 100, 150, 200, 300, 500] #[int(x) for x in np.linspace(start=10, stop=2000, num=10)]
        learning_rate = [0.0001, 0.001, 0.01, 0.1]
        subsample = [0.5, 0.7, 1.0]
        max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        param_grid_gb = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "max_depth": max_depth,
        }

        gbc = GradientBoostingClassifier(random_state=42)

        rs_gb = RandomizedSearchCV(
            estimator=gbc,
            param_distributions=param_grid_gb,
            scoring=None,
            refit='f1',
            n_iter=10,
            return_train_score=True,
            cv=None,
            n_jobs=-1,
            verbose=1
        )

        # Fit
        gb_train = rs_gb.fit(self.X_train, self.y_train)
        best_gb = rs_gb.best_estimator_
        best_gb_index = rs_gb.best_index_
        print("Best params: n_estimators =", best_gb.n_estimators, ", learning_rate =", best_gb.learning_rate, ", subsample =", best_gb.subsample, ", max_depth =", best_gb.max_depth)
        return best_gb
    
    def make_predictions(self, best_gb):
        y_pred_gbc = best_gb.predict(self.X_test)
        accuracy_gbc = accuracy_score(self.y_test, y_pred_gbc)
        print(f"Accuracy: {accuracy_gbc * 100:.2f}%")
        return accuracy_gbc, y_pred_gbc





def main(path):
    """Main function to initialize and process data."""
    model = XGBoost_predictions(path)
    
    #print("Cleaning data...")
    #model.clean_data()
    
    print("Encoding target...")
    model.encode_target()

    print("Splitting data...")
    #X_train, X_test, y_train, y_test = model.splitting()

    print("Finding optimal hyperparameters...")
    best_gb = model.randomizedSearch()

    print("Making predictions...")
    accuracy_gbc, y_pred_gbc = model.make_predictions(best_gb)

    return model, accuracy_gbc, y_pred_gbc  # Returning the model in case you need it in Jupyter Notebook

if __name__ == "__main__":
    main()  # Run only when executed as a script
