from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    error_term = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error_term

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data points')
    plt.plot(X, model.predict(X_reshaped), color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter plot of X vs Y with fitted regression line')
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.uniform(0, 1, N)
        error_term_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_term_sim

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model.fit(X_sim_reshaped, Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # TODO 8: Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(slopes, bins=30, color='blue', alpha=0.7)
    plt.axvline(slope, color='red', linestyle='dashed', linewidth=2, label='Observed slope')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title('Histogram of slopes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(intercepts, bins=30, color='green', alpha=0.7)
    plt.axvline(intercept, color='red', linestyle='dashed', linewidth=2, label='Observed intercept')
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.title('Histogram of intercepts')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_diffs = np.abs(np.array(slopes) - beta1)
    observed_slope_diff = np.abs(slope - beta1)
    slope_more_extreme = np.mean(slope_diffs >= observed_slope_diff)

    intercept_diffs = np.abs(np.array(intercepts) - beta0)
    observed_intercept_diff = np.abs(intercept - beta0)
    intercept_extreme = np.mean(intercept_diffs >= observed_intercept_diff)

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # TODO 10: Calculate p-value based on test type
    if test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "less":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "two-sided":
        p_value = np.mean(
            np.abs(simulated_stats - hypothesized_value)
            >= np.abs(observed_stat - hypothesized_value)
        )

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value <= 0.0001:
        fun_message = "...?!"
    else:
        fun_message = None

    # TODO 12: Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=30, color='blue', alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')

    if test_type == 'greater':
        plt.fill_betweenx(
            [0, plt.gca().get_ylim()[1]],
            observed_stat,
            plt.gca().get_xlim()[1],
            color='red',
            alpha=0.2,
        )
    elif test_type == 'less':
        plt.fill_betweenx(
            [0, plt.gca().get_ylim()[1]],
            plt.gca().get_xlim()[0],
            observed_stat,
            color='red',
            alpha=0.2,
        )
    elif test_type == 'two-sided':
        left = hypothesized_value - np.abs(observed_stat - hypothesized_value)
        right = hypothesized_value + np.abs(observed_stat - hypothesized_value)
        plt.fill_betweenx(
            [0, plt.gca().get_ylim()[1]],
            plt.gca().get_xlim()[0],
            left,
            color='red',
            alpha=0.2,
        )
        plt.fill_betweenx(
            [0, plt.gca().get_ylim()[1]],
            right,
            plt.gca().get_xlim()[1],
            color='red',
            alpha=0.2,
        )

    plt.xlabel('Simulated Statistics')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated Statistics')
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # TODO 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level
    alpha = 1 - confidence_level / 100
    degrees_of_freedom = S - 1
    t_crit = t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    ci_lower = mean_estimate - t_crit * std_estimate / np.sqrt(S)
    ci_upper = mean_estimate + t_crit * std_estimate / np.sqrt(S)

    # TODO 16: Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # TODO 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 2))
    plt.scatter(estimates, np.zeros_like(estimates), color='gray', alpha=0.5, label='Estimates')

    mean_color = 'green' if includes_true else 'red'
    plt.scatter(mean_estimate, 0, color=mean_color, label='Mean Estimate')

    plt.hlines(0, ci_lower, ci_upper, colors=mean_color, linewidth=4, label='Confidence Interval')

    plt.axvline(true_param, color='blue', linestyle='dashed', label='True Parameter')

    plt.xlabel('Estimate Value')
    plt.yticks([])
    plt.title(f'Confidence Interval for {parameter}')
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
