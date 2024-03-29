# National Health and Nutrition Examination Survey (NHANES)

Here we demonstrate EARTH by building regression models
to study the variation in systolic blood pressure (SBP)
in terms of age, BMI, sex, and ethnicity.

See the [NHANES website](https://wwwn.cdc.gov/nchs/nhanes) for more
information about these data.

````julia
using CategoricalArrays
using CSV
using DataFrames
using Downloads
using Earth
using Plots
using ReadStatTables
using Statistics

dfile = "assets/nhanes2017.csv.gz";
````

The function below downloads and merges the data sets.

````julia
function get_data()

    mkpath("assets")
    y = 2017
    letter = "ABCDEFGHIJ"[1 + div(y - 1999, 2)]

    dx = []
    for f in ["DEMO", "BMX", "BPX"]
        g = "$(f)_$(letter).XPT"
        s = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/$(g)"
        h = joinpath("assets", g)
        Downloads.download(s, h)
        da = readstat(h) |> DataFrame
        push!(dx, da)
    end

    demog = dx[1][:, [:SEQN, :RIAGENDR, :RIDAGEYR, :RIDRETH1]]
    bmx = dx[2][:, [:SEQN, :BMXBMI]]
    sbp = dx[3][:, [:SEQN, :BPXSY1]]

    da = leftjoin(demog, bmx, on=:SEQN)
    da = leftjoin(da, sbp, on=:SEQN)

    da = da[completecases(da), :]
    da = disallowmissing(da)
    da[!, :RIAGENDR] = replace(da[:, :RIAGENDR], 1=>"Male", 2=>"Female") .|> String
    da[!, :RIDRETH1] = replace(da[:, :RIDRETH1], 1=>"MA", 2=>"OH", 3=>"NHW", 4=>"NHB", 5=>"OR") .|> String
    da = filter(r->r.RIDAGEYR >= 18, da)

    CSV.write(dfile, da; compress=true)
end;
````

Download the data only if it is not already present

````julia
isfile(dfile) || get_data();
````

Read the data into a data frame.

````julia
da = open(dfile) do io
    CSV.read(io, DataFrame)
end;
````

To use categorical variables in Earth they must be
explicitly typed as CategoricalArray.

````julia
da[!, :RIDRETH1] = CategoricalArray(da[:, :RIDRETH1]);
da[!, :RIAGENDR] = CategoricalArray(da[:, :RIAGENDR]);
````

Define the response variable as a float vector:

````julia
y = da[:, :BPXSY1];
y = (y .- mean(y)) ./ std(y);
````

Construct the covariates as a named tuple:

````julia
X = (RIDAGEYR=da[:, :RIDAGEYR], BMXBMI=da[:, :BMXBMI], RIAGENDR=da[:, :RIAGENDR], RIDRETH1=da[:, :RIDRETH1]);
````

Fit an additive model, limiting the order and degree of each
term to 1.  Note that each term only involves a single covariate.

````julia
cfg = EarthConfig(; maxorder=1, maxdegree=1)
m1 = fit(EarthModel, X, y; config=cfg, verbosity=1)
````

````
     Coef    Std coef    Term
    -0.240       --      intercept
     0.567       0.477   intercept * h(RIDAGEYR - -0.831)
    -0.171      -0.042   intercept * h(-0.831 - RIDAGEYR)
     0.165       0.092   intercept * h(BMXBMI - 0.473)
    -0.082      -0.051   intercept * h(0.473 - BMXBMI)
    -0.077      -0.077   intercept * h(1.839 - RIDRETH1::NHB)
    -0.072      -0.072   intercept * h(RIDRETH1::NHW - -0.730)
    -0.063      -0.063   intercept * h(RIAGENDR::Female - -1.021)

````

Fit another model that allows nonlinear main effects and two-way
interactions.

````julia
cfg = EarthConfig(; maxorder=2, maxdegree=1)
m2 = fit(EarthModel, X, y; config=cfg, verbosity=1)
````

````
     Coef    Std coef    Term
    -0.161       --      intercept
     0.153       0.129   intercept * h(RIDAGEYR - -0.831)
    -0.293      -0.072   intercept * h(-0.831 - RIDAGEYR)
     0.140       0.078   intercept * h(BMXBMI - 0.473)
    -0.316      -0.195   intercept * h(0.473 - BMXBMI)
     0.073       0.116   intercept * h(RIDAGEYR - -0.831) * h(1.370 - RIDRETH1::NHW)
     0.301       0.231   intercept * h(0.473 - BMXBMI) * h(RIDAGEYR - -0.455)
     0.160       0.078   intercept * h(0.473 - BMXBMI) * h(-0.455 - RIDAGEYR)
     0.146       0.146   intercept * h(RIDRETH1::NHB - -0.544)
    -0.133      -0.070   intercept * h(RIDRETH1::NHB - -0.544) * h(RIDAGEYR - 0.323)
    -0.055      -0.053   intercept * h(RIDRETH1::NHB - -0.544) * h(0.323 - RIDAGEYR)
     0.169       0.256   intercept * h(RIDAGEYR - -0.831) * h(RIAGENDR::Female - -1.021)
    -0.223      -0.223   intercept * h(RIAGENDR::Female - -1.021)
     0.073       0.032   intercept * h(BMXBMI - 0.473) * h(RIDRETH1::OH - -0.316)

````

Get the adjusted r-squared sequences for each model.

````julia
r2_1 = gr2(m1)
r2_2 = gr2(m2)
p = plot(m1.nterms, r2_1, xlabel="Number of terms", ylabel="R2", label="maxorder=1")
plot!(p, m2.nterms, r2_2, label="maxorder=2")
Plots.savefig(p, "./assets/nhanes1.svg");
````

![R-squares](assets/nhanes1.svg)

The function below generates the fitted mean blood pressure
at fixed levels of sex, BMI, and race.

````julia
function sbp_by_age(m; sex="Female", bmi=25, eth="NHB")
    dp = da[1:100, [:RIDAGEYR, :BMXBMI, :RIAGENDR, :RIDRETH1]]
    dp[:, :RIDAGEYR] = range(18, 80, 100)
    dp[:, :BMXBMI] .= bmi
    dp[:, :RIAGENDR] .= sex
    dp[:, :RIDRETH1] .= eth
    yh = predict(m, dp)
    return dp[:, :RIDAGEYR], yh
end;
````

The plot below shows the estimated conditional mean blood
pressure values for non-hispanic black females, at three
levels of BMI, in an additive model.

````julia
age, sbp = sbp_by_age(m1; bmi=25)
p = plot(age, sbp, xlabel="Age", ylabel="SBP", label="BMI=25")
age, sbp = sbp_by_age(m1; bmi=30)
plot!(p, age, sbp, label="BMI=30")
age, sbp = sbp_by_age(m1; bmi=35)
plot!(p, age, sbp, label="BMI=35")
Plots.savefig(p, "./assets/nhanes2.svg");
````

![Fitted means](assets/nhanes2.svg)

Below we perform the same analysis as above, but here allowing two-way interactions.

````julia
age, sbp = sbp_by_age(m2; bmi=25)
p = plot(age, sbp, xlabel="Age", ylabel="SBP", label="BMI=25")
age, sbp = sbp_by_age(m2; bmi=30)
plot!(p, age, sbp, label="BMI=30")
age, sbp = sbp_by_age(m2; bmi=35)
plot!(p, age, sbp, label="BMI=35")
Plots.savefig(p, "./assets/nhanes3.svg");
````

![Fitted means](assets/nhanes3.svg)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

