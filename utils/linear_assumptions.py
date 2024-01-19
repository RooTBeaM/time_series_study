# References to https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def linear_regression_assumptions(features, label, feature_names=None):
    """
    Tests a linear regression on the model to see if assumptions are being met
    """
    # Setting feature names to x1, x2, x3, etc. if they are not defined
    if feature_names is None:
        feature_names = ['X'+str(feature+1) for feature in range(features.shape[1])]
    
    print('Fitting linear regression')
    # Multi-threading if the dataset is a size where doing so is beneficial
    if features.shape[0] < 100000:
        model = LinearRegression(n_jobs=-1)
    else:
        model = LinearRegression()
        
    model.fit(features, label)
    
    # Returning linear regression R^2 and coefficients before performing diagnostics
    r2 = model.score(features, label)
    print()
    print('R^2:', r2, '\n')
    print('Coefficients')
    print('-------------------------------------')
    print('Intercept:', model.intercept_)
    
    for feature in range(len(model.coef_)):
        print('{0}: {1}'.format(feature_names[feature], model.coef_[feature]))

    print('\n ====== Performing linear regression assumption testing ======')
    
    # Creating predictions and calculating residuals for assumption tests
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    def calculate_residuals(model, features, label):
        """
        Creates predictions on the features with the model and calculates residuals
        """
        predictions = model.predict(features)
        df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
        df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
        
        return df_results

    def linear_assumption(model, features, label):
        """
        Linearity: Assumes that there is a linear relationship between the predictors and
                the response variable. If not, either a quadratic term or another
                algorithm should be used.
        """
        print('>>> Assumption 1: Linear Relationship between the Target and the Feature')
            
        print('Checking with a scatter plot of actual vs. predicted.',
            'Predictions should follow the diagonal line.')
        
        # Calculating residuals for the plot
        df_results = calculate_residuals(model, features, label)
        
        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, height=3, aspect=1.5)
            
        # Plotting the diagonal line
        line_coords = np.arange(df_results.min().min(), df_results.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()

    def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.
                
        This assumption being violated primarily causes issues with the confidence intervals
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('>>> Assumption 2: The error terms are normally distributed')
        
        # Calculating residuals for the Anderson-Darling test
        df_results = calculate_residuals(model, features, label)
        
        print('Using the Anderson-Darling test for normal distribution')

        # Performing the test on the residuals
        p_value = normal_ad(df_results['Residuals'])[1]
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
        
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Residuals are not normally distributed')
        else:
            print('Residuals are normally distributed')
        
        # Plotting the residuals distribution
        plt.subplots(figsize=(6, 3))
        plt.title('Distribution of Residuals')
        sns.histplot(df_results['Residuals'])
        plt.show()
        if p_value > p_value_thresh:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
            print()
            print('Confidence intervals will likely be affected')
            print('Try performing nonlinear transformations on variables')
        print()

    def multicollinearity_assumption(model, features, label, feature_names=None):
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                        correlation among the predictors, then either remove prepdictors with high
                        Variance Inflation Factor (VIF) values or perform dimensionality reduction
                            
                        This assumption being violated causes issues with interpretability of the 
                        coefficients and the standard errors of the coefficients.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        print('>>> Assumption 3: Little to no multicollinearity among predictors')
            
        # Plotting the heatmap
        plt.figure(figsize = (10,8))
        sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=False)
        plt.title('Correlation of Variables')
        plt.show()
            
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
        
        # Gathering the VIF for each variable
        VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
        for idx, vif in enumerate(VIF):
            print('{0}: {1}'.format(feature_names[idx], vif))
            
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
        definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
        print()
        print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
        print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
        print()

        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied!')
            else:
                print('***Assumption possibly satisfied')
                print('Coefficient interpretability may be problematic')
                print('Consider removing variables with a high Variance Inflation Factor (VIF)')

        else:
            print('*** Assumption not satisfied ***')
            print('Coefficient interpretability will be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')
        print()

    def autocorrelation_assumption(model, features, label):
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                        autocorrelation, then there is a pattern that is not explained due to
                        the current value being dependent on the previous value.
                        This may be resolved by adding a lag variable of either the dependent
                        variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson
        print('>>> Assumption 4: No Autocorrelation')
        
        # Calculating residuals for the Durbin Watson-tests
        df_results = calculate_residuals(model, features, label)

        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        durbinWatson = durbin_watson(df_results['Residuals'])
        print('Durbin-Watson:', durbinWatson)
        if durbinWatson < 1.5:
            print('Signs of positive autocorrelation')
            print('Assumption not satisfied')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation')
            print('Assumption not satisfied')
        else:
            print('Little to no autocorrelation')
            print('Assumption satisfied')
        print('-------------------------------------')

    def homoscedasticity_assumption(model, features, label):
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('>>> Assumption 5: Homoscedasticity of Error Terms')
        print('Residuals should have relative constant variance')
            
        # Calculating residuals for the plot
        df_results = calculate_residuals(model, features, label)

        # Plotting the residuals
        plt.subplots(figsize=(6, 3))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show()  

    linear_assumption(model, features, label)
    normal_errors_assumption(model, features, label)
    multicollinearity_assumption(model, features, label, features.columns)
    autocorrelation_assumption(model, features, label)
    homoscedasticity_assumption(model, features, label)