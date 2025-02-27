import ROOT
import matplotlib.pyplot as plt
from ROOT import RooFit, RooRealVar, RooDataHist, RooArgList, RooArgSet, RooGaussian, RooChebychev, RooAddPdf, RooStats
# Open the ROOT file
file = ROOT.TFile.Open("InvariantMass_0_50_4.5_pT_8.0_run_342361.root")
hMass = file.Get("hMass_0_50_PtRange_4.5_8.0")
hMass.SetDirectory(0)
file.Close()

# Define the invariant mass variable
mass = RooRealVar("mass", "Invariant Mass", hMass.GetXaxis().GetXmin(), hMass.GetXaxis().GetXmax(), "GeV/c^{2}")
# Create a RooDataHist from the histogram
data = RooDataHist("data", "Dataset with mass", RooArgList(mass), hMass)


m = 2.99001
s = 0.0014145
mean = RooRealVar("mean", "Mean of Gaussian", m, m-s, m+s)
sigma = RooRealVar("sigma", "Width of Gaussian", s, -100*s, 100*s)
gauss = RooGaussian("gauss", "Gaussian Signal", mass, mean, sigma)

mean.setConstant(True)
sigma.setConstant(True)


# Define the background polynomial (2nd order)
a0 = RooRealVar("a0", "a0",  -10, 10)  # Valores menores
a1 = RooRealVar("a1", "a1",  -10, 10)
background = RooChebychev("background", "Polynomial Background", mass, RooArgList(a0, a1))

# a0.setConstant(True)
# a1.setConstant(True)
# a2.setConstant(True)

# Define the fractions of signal and background
nsig = RooRealVar("nsig", "Number of signal events", 100, 0, 1000000)
nbkg = RooRealVar("nbkg", "Number of background events", 1000, 0, 1000000)

# Create the total model
model = RooAddPdf("model", "Signal + Background", RooArgList(gauss, background), RooArgList(nsig, nbkg))



# Plot the data and the fit result
frame = mass.frame()
data.plotOn(frame, RooFit.Name("data"))
model.plotOn(frame, RooFit.Name("model"))
model.plotOn(frame, RooFit.Name("background"), RooFit.Components("background"), RooFit.LineColor(ROOT.kCyan), RooFit.LineStyle(ROOT.kDashed), RooFit.LineWidth(3))
model.plotOn(frame, RooFit.Name("gauss"), RooFit.Components("gauss"), RooFit.LineColor(ROOT.kRed-4), RooFit.LineStyle(ROOT.kSolid), RooFit.LineWidth(3))
model.paramOn(frame)


# Create a workspace
workspace = ROOT.RooWorkspace("workspace")
getattr(workspace, 'import')(data)
getattr(workspace, 'import')(model)

workspace.Print()
# Set the confidence level
confidence_level = 0.95

# Create a ModelConfig
model_config = RooStats.ModelConfig("model_config", workspace)
model_config.SetPdf(model)
model_config.SetParametersOfInterest(nsig)
model_config.SetObservables(RooArgSet(mass))
model_config.SetNuisanceParameters(RooArgSet(a0, a1, nbkg))
oldValue = nsig.getVal()

# Create a null (background-only) model
null_model = RooStats.ModelConfig("null_model", workspace)
null_model.SetPdf(model)
nsig.setVal(0)
workspace.var("nsig").setVal(0)
null_model.SetParametersOfInterest(nsig)
null_model.SetObservables(RooArgSet(mass))
null_model.SetNuisanceParameters(RooArgSet(a0, a1, nbkg))
null_model.SetSnapshot(RooArgSet(nsig))
null_model.GetParametersOfInterest().first().setVal(oldValue)
print("Null model snapshot:")
null_model.GetSnapshot().Print("v")

# Perform the upper limit calculation
calculator = RooStats.AsymptoticCalculator(data, null_model, model_config)
calculator.SetOneSided(True)
hypo_test_result = calculator.GetHypoTest()
if not hypo_test_result:
    print("Erro: HypoTest result is None!")

# Extract the p-value
invereter = RooStats.HypoTestInverter(calculator)
invereter.SetConfidenceLevel(confidence_level)
invereter.UseCLs(True)
invereter.SetVerbose(True)
#invereter.SetFixedScan(2, 22920, 22930)  # Reduzindo o range do scan 0.050321
# invereter.SetFixedScan(1000, 0, 22920)

#invereter.SetAutoScan()
result = invereter.GetInterval()

# Print the result
upper_limit = result.UpperLimit()
low_limit = result.LowerLimit()
error_up = result.UpperLimitEstimatedError()
error_lower = result.LowerLimitEstimatedError()

result = model.fitTo(data, RooFit.Save(), RooFit.Strategy(1))

## set signal to 0
nsig.setVal(0)
# Plot the data and the fit result
frame_signal_null = mass.frame()
frame_signal_null.SetName("frame_signal_null")
model.plotOn(frame_signal_null, RooFit.Name("model"))

## set signal to upper limit
nsig.setVal(upper_limit)
frame_signal = mass.frame()
frame_signal.SetName("frame_signal")
model.plotOn(frame_signal, RooFit.Name("model"))


# Plot the data and the fit result
# Fit the model to the data
frame = mass.frame()
frame.SetName("frame_data")
data.plotOn(frame, RooFit.Name("data"))
model.plotOn(frame, RooFit.Name("model"))
model.plotOn(frame, RooFit.Name("background"), RooFit.Components("background"), RooFit.LineColor(ROOT.kCyan), RooFit.LineStyle(ROOT.kDashed), RooFit.LineWidth(3))
model.plotOn(frame, RooFit.Name("gauss"), RooFit.Components("gauss"), RooFit.LineColor(ROOT.kRed-4), RooFit.LineStyle(ROOT.kSolid), RooFit.LineWidth(3))
model.paramOn(frame)


## save to file
file = ROOT.TFile("data_signal.root", "RECREATE")
frame_signal.Write()
frame_signal_null.Write()
frame.Write()
file.Close()

print("-------------------------------------------------------------------------------")
print(f"Upper limit at {confidence_level*100:.0f}% CL: {upper_limit:.2f} +- {error_up}")


# nsig_values = []
# nsig_values.append(upper_limit)
# nll_values = []

# # Construir a função de máxima verossimilhança (NLL)
# nll = model.createNLL(data, RooFit.NumCPU(2))

# # Minimizador para encontrar melhor valor antes do scan
# minimizer = ROOT.RooMinimizer(nll)
# minimizer.migrad()

# # Executar o scan para cada valor de nsig
# for i in range(100):  # 10 iterações do scan
#     # Avliar NLL para este valor de nsig
#     likelihood_val = nll.getVal()
#     nll_values.append(likelihood_val)

# # Normalizar NLL para que o mínimo seja zero (-2ΔlnL)
# nll_min = min(nll_values)
# delta_nll_values = [2 * (val - nll_min) for val in nll_values]

# # Criar o gráfico do Likelihood Scan
# plt.figure(figsize=(8, 6))
# plt.plot(nsig_values, delta_nll_values, marker='o', linestyle='-', color='red', label=r'$-2\Delta \ln L$')
# plt.axhline(y=1.0, color='blue', linestyle='dashed', label=r'$1\sigma$ (68%)')
# plt.axhline(y=3.84, color='green', linestyle='dashed', label=r'$95\%$ CL')

# plt.xlabel("nsig")
# plt.ylabel(r"$-2\Delta \ln L$")
# plt.title("Likelihood Scan for nsig")
# plt.legend()
# plt.grid()

# # Salvar a imagem do gráfico
# plt.savefig("likelihood_scan_nsig.png")
# plt.show()