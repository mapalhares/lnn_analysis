# Read a AO2D data and json
import ROOT
import numpy as np
import uproot as up
import pandas as pd
import argparse
import os
import yaml
from hipe4ml.tree_handler import TreeHandler
import utils_Eff as utils
ROOT.gROOT.SetBatch(True)


parser = argparse.ArgumentParser(description='Configure the parameters of the script.')
parser.add_argument('--config-file', dest='config_file', help='path to the YAML file with configuration.', default='')
args = parser.parse_args()

# initialise parameters from parser (can be overwritten by external yaml file)

if args.config_file == '':
    print('** No config file provided. Exiting. **')
    exit()

config_file = open(args.config_file, 'r')
config = yaml.full_load(config_file)
# mc = config['mc']

input_files_name = config['input_files']
output_dir_name = config['output_dir']
output_file_name = config['output_file']
tree_names = config['tree_names']
selections = config['selection']
selections_string = utils.convert_sel_to_string(selections)


print('**********************************')
print('Running LnnTrees_analysis.py')
print('**********************************')


#Lnn info 
hCentrality = ROOT.TH1F("hCentrality","hCentrality", 100, 0., 100.)
hCentrality_Rec = ROOT.TH1F("hCentrality","hCentrality", 100, 0., 100.)

hPtReco = ROOT.TH1F("hPtReco", "LnnPt", 20, 0., 10.)
hPtReco_1 = ROOT.TH1F("hPtReco", "LnnPt", 20, 0., 10)
hPtAccepted = ROOT.TH1F("hPtAccepted", "Distribuição de pT dos Eventos Aceitos", 16, 0., 8.)
hPtRecoIsMatter = ROOT.TH1F("hPtRecoIsMatter", "MatterLnnPt", 16, 0., 8.)
hPtRecoIsAntiMatter = ROOT.TH1F("hPtRecoIsAntiMatter", "AntiLnnPt", 16, 0., 8.)

hPtReco_by_0_10 = ROOT.TH1F("hPtReco_by_0_10", "RecoPt - 0-10%", 20, 0., 10.)
hPtReco_by_0_10_dfReco = ROOT.TH1F("hPtReco_by_0_10_dfReco", "RecoPt - 0-10%", 20, 0., 10.)
hPtGen_by_0_10 = ROOT.TH1F('hPtGen_by_0_10', "GenPt - 0-10%", 20, 0., 10.)
hPtGen_by_0_10_only_cent = ROOT.TH1F('hPtGen_by_0_10_only_cent', "GenPt - 0-10%", 20, 0., 10.)

    
hPtReco_by_10_30 = ROOT.TH1F("hPtReco_by_10_30", "RecoPt - 10-30%", 20, 0., 10.)
hPtReco_by_10_30_dfReco = ROOT.TH1F("hPtReco_by_10_30_dfReco", "RecoPt - 10-30%", 20, 0., 10.)
hPtGen_by_10_30 = ROOT.TH1F('hPtGen_by_10_30', "GenPt - 10-30%", 20, 0., 10.)
hPtGen_by_10_30_only_cent = ROOT.TH1F('hPtGen_by_10_30_only_cent', "GenPt - 10-30%", 20, 0., 10.)
    
hPtReco_by_30_50 = ROOT.TH1F("hPtReco_by_30_50", "RecoPt - 30-50%", 20, 0., 10.)
hPtReco_by_30_50_dfReco = ROOT.TH1F("hPtReco_by_30_50_dfReco", "RecoPt - 30-50%", 20, 0., 10.)
hPtGen_by_30_50 = ROOT.TH1F('hPtGen_by_30_50', "GenPt - 30-50%", 20, 0., 10.)
hPtGen_by_30_50_only_cent = ROOT.TH1F('hPtGen_by_30_50_only_cent', "GenPt - 30-50%", 20, 0., 10.)


hPtGen = ROOT.TH1F('hPtGen', "LnnGenPt", 20, 0., 10.)
hPtGen_1 = ROOT.TH1F('hPtGen', "LnnGenPt", 20, 0., 10.)
hPtGenIsMatter = ROOT.TH1F('hPtGenIsMatter', "MatterLnnPtGen", 100, 0., 10.)
hPtGenIsAntiMatter = ROOT.TH1F('hPtGenIsAntiMatter', "AntiMatterPtGen", 100, 0., 10.)

hPtReco3H = ROOT.TH1F("hPtReco3H", "Pt3H", 16, 0., 8.)
hPtGen3H = ROOT.TH1F("hPtGen3H", "Pt3HGen", 16, 0., 8.)

hCtReco = ROOT.TH1F("hCt","Ct", 16, 0., 8.)
hCtGen = ROOT.TH1F("hCtGen", "GenCt", 50, 0., 40)

hDecayLen = ROOT.TH1F("hDecayLen", "hDecayLen", 15, 0., 40.)

hIsMatter = ROOT.TH1F("hIsMatter", "IsMatter",  2, 0., 2.)
hIsReco = ROOT.TH1F("hIsReco", "IsReco", 2, 0., 2.)
hIsSignal = ROOT.TH1F("hIsSignal", "hIsSignal", 2, 0., 2.)
hIsSuvEvSel8 = ROOT.TH1F("hIsSuvEvSel8", "IsSuvEvSel8",  2, 0., 2.)

hMassLnn = ROOT.TH1F("hMassLnn", "Invariant Mass", 40, 2.98, 3.02)
hGenPz = ROOT.TH1F('hGenPz', "hGenPz", 10, -10, 10)
hGenP = ROOT.TH1F('hGenP', "hGenP", 10, 0, 10)
hGenDecLen = ROOT.TH1F("hGenDecLen", "GenDecLen", 15, 0., 40.)
h2ResolutionPtvsPt = ROOT.TH2F("hResolutionPtvsPt", "ResolutionPtvsPt", 50, 0, 10., 50, -0.2, 0.2)
hDecayLen = ROOT.TH1F("hDecayLen", "hDecayLen", 15, 0., 40.)

#hNSigma3H = ROOT.TH1F("hNSigma3H", "nSigma3H", 30, -3, 3)
h2CentVsPtReco = ROOT.TH2F("h2CentVsPtReco", "h2CentVsPtReco", 100, 0, 100, 20, 0, 10)
h2CentVsPtGen = ROOT.TH2F("h2CentVsPtGen", "h2CentVsPtGen", 100, 0, 100, 20, 0, 10)

hnTPCClus3H = ROOT.TH1F("nTPCClus3H", "nTPCClus3H", 10, 60, 160)

hCosPA = ROOT.TH1F('hCosPA','CosPA', 50, 0.95, 1.)

# DATA FRAME #
hdl = TreeHandler(input_files_name, tree_names, folder_name='DF*')
df_lnnInfo = hdl.get_data_frame()

df, IsGen_df =  utils.correct_and_convert_df(df_lnnInfo)

spectra_file = ROOT.TFile.Open('utils/fCombineTritonSpecBWFit_0-100.root')
h3_spectrum = spectra_file.Get('TritonBW_0-90')
spectra_file.Close()

spectra_file_3HL = ROOT.TFile.Open('utils/H3L_BWFit.root')
h3L_0_10 = spectra_file_3HL.Get('BlastWave_H3L_0_10')
h3L_10_30 = spectra_file_3HL.Get('BlastWave_H3L_10_30')
h3L_30_50 = spectra_file_3HL.Get('BlastWave_H3L_30_50')
spectra_file_3HL.Close()

utils.reweight_pt_spectrum(df, 'fAbsGenPt', h3_spectrum)
utils.reweight_pt_spectrum_0_10(df, 'fAbsGenPt', h3L_0_10)
utils.reweight_pt_spectrum_10_30(df, 'fAbsGenPt', h3L_10_30)
utils.reweight_pt_spectrum_30_50(df, 'fAbsGenPt', h3L_30_50)

# fill histograms to be put at denominator of efficiency
utils.fill_th1f_hist(hPtGen, IsGen_df, 'fAbsGenPt')
utils.fill_th1f_hist(hCtGen, IsGen_df, 'fGenCt')
utils.fill_th1f_hist(hGenDecLen, IsGen_df, 'fGenDecLen')
utils.fill_th1f_bool_hist(hGenPz, IsGen_df, 'fGenPz')
utils.fill_th1f_bool_hist(hGenP, IsGen_df, 'fGenP')


utils.number_of_gen_by_centrality(IsGen_df, 'fCentralityFT0C', 'fIsSignal', 'Number_of_Gen_WithOutRW.root')

df_gen_cent_0_10 = IsGen_df[(IsGen_df['fCentralityFT0C'] >=0) & (IsGen_df['fCentralityFT0C'] < 10)] 
df_gen_cent_10_30 = IsGen_df[(IsGen_df['fCentralityFT0C'] >=10) & (IsGen_df['fCentralityFT0C'] < 30)] 
df_gen_cent_30_50 = IsGen_df[(IsGen_df['fCentralityFT0C'] >=30) & (IsGen_df['fCentralityFT0C'] < 50)] 

utils.fill_th1f_hist(hPtGen_by_0_10_only_cent, df_gen_cent_0_10, 'fAbsGenPt')
utils.fill_th1f_hist(hPtGen_by_10_30_only_cent, df_gen_cent_10_30, 'fAbsGenPt')
utils.fill_th1f_hist(hPtGen_by_30_50_only_cent, df_gen_cent_30_50, 'fAbsGenPt')

utils.number_of_lnn_by_pT(df_gen_cent_0_10, 'fIsSignal', 'fAbsGenPt', 'fSurvivedEventSelection', 'number_gen_cent_lnn_0_10.root')
utils.number_of_lnn_by_pT(df_gen_cent_10_30, 'fIsSignal', 'fAbsGenPt', 'fSurvivedEventSelection', 'number_gen_cent_lnn_10_30.root')
utils.number_of_lnn_by_pT(df_gen_cent_30_50, 'fIsSignal', 'fAbsGenPt', 'fSurvivedEventSelection', 'number_gen_cent_lnn_30_50.root')

df_accepted = df[(df['rej'] == 1)]

#Added the (df['fLnnRapidity'] < 0.5) & (df['fIsSignal'] == 1)  as same condtion in the IsGen_df
df_accepted_0_10 =  df[(df['rej_0_10'] == 1) & (df['fLnnRapidity'] < 0.5) & (df['fIsSignal'] == 1) & (df['fSurvivedEventSelection'] ==1) & (df['fCentralityFT0C'] >=0) & (df['fCentralityFT0C'] <= 10) ]
df_accepted_10_30 = df[ (df['rej_10_30'] == 1)  & (df['fLnnRapidity'] < 0.5) & (df['fIsSignal'] == 1) & (df['fSurvivedEventSelection'] ==1) & (df['fCentralityFT0C'] >=10) & (df['fCentralityFT0C'] <=30)]
df_accepted_30_50 = df[(df['rej_30_50'] == 1) & (df['fLnnRapidity'] < 0.5) & (df['fIsSignal'] == 1) & (df['fSurvivedEventSelection'] ==1) & (df['fCentralityFT0C'] >=30) & (df['fCentralityFT0C'] <=50)]


utils.number_of_gen_by_centrality_RW(df_accepted_0_10, df_accepted_10_30, df_accepted_30_50, 'fIsSignal',  'Number_of_Gen_RW.root')

utils.fill_th1f_bool_hist(hIsSuvEvSel8, IsGen_df, 'fSurvivedEventSelection')

utils.fill_th1f_hist(hPtGen_1, df_accepted, 'fAbsGenPt')
utils.fill_th1f_hist(hPtGen_by_0_10, df_accepted_0_10, 'fAbsGenPt')
utils.fill_th1f_hist(hPtGen_by_10_30, df_accepted_10_30, 'fAbsGenPt')
utils.fill_th1f_hist(hPtGen_by_30_50, df_accepted_30_50, 'fAbsGenPt')

utils.fill_th1f_hist(hCentrality, df, 'fCentralityFT0C')
utils.fill_th1f_bool_hist(hIsReco, df, 'fIsReco')
utils.fill_th1f_bool_hist(hIsSignal, df, 'fIsSignal')

utils.number_of_reco_by_centrality(IsGen_df, 'fCentralityFT0C','fIsReco', 'Number_of_Reco_BWSel_lnn__per_centrality.root')

utils.number_of_lnn_by_pT(df_accepted_0_10, 'fIsSignal', 'fAbsGenPt', 'fSurvivedEventSelection', 'number_gen_lnn_0_10.root')
utils.number_of_lnn_by_pT(df_accepted_0_10, 'fIsSignal', 'fAbsGenPt', 'fSurvivedEventSelection', 'number_gen_lnn_10_30.root')
utils.number_of_lnn_by_pT(df_accepted_0_10, 'fIsSignal', 'fAbsGenPt', 'fSurvivedEventSelection', 'number_gen_lnn_30_50.root')
    
############# Common filtering #############
if selections_string != '':
    df_filtered = df.query(selections_string)
    
hNSigma3H = utils.computeNSigma3H(df, 'fTPCmom3H')

df_reco_0_10_wth_rej = df_filtered[(df['fCentralityFT0C'] >= 0) & (df_filtered['fSurvivedEventSelection'] ==1) & (df_filtered['fCentralityFT0C'] <= 10)]
df_reco_10_30_wth_rej= df_filtered[(df_filtered['fCentralityFT0C'] >=10) & (df_filtered['fSurvivedEventSelection'] ==1) & (df_filtered['fCentralityFT0C'] <=30)]
df_reco_30_50_wth_rej = df_filtered[(df_filtered['fCentralityFT0C'] >=30) & (df_filtered['fSurvivedEventSelection'] ==1) & (df_filtered['fCentralityFT0C'] <=50)]
    
df_reco_0_10= df_filtered[(df_filtered['fCentralityFT0C'] >= 0) & (df_filtered['fSurvivedEventSelection'] ==1) & (df_filtered['fCentralityFT0C'] <= 10) & (df_filtered['rej_0_10'] == 1)]
df_reco_10_30 = df_filtered[(df_filtered['fCentralityFT0C'] >=10) & (df_filtered['fSurvivedEventSelection'] ==1) & (df['fCentralityFT0C'] <=30) & (df_filtered['rej_10_30'] == 1)]
df_reco_30_50 = df_filtered[(df_filtered['fCentralityFT0C'] >=30) & (df_filtered['fCentralityFT0C'] <=50) & (df_filtered['fSurvivedEventSelection'] ==1) & (df_filtered['rej_30_50'] == 1)]

utils.number_of_reco_by_centrality(df_filtered, 'fCentralityFT0C','fIsReco', 'Number_of_Reco_AFSel_lnn__per_centrality.root')
utils.number_of_reco_by_centrality_RW(df_reco_0_10, df_reco_10_30, df_reco_30_50, 'fIsReco', 'Number_of_Reco_Sel_RW_per_centrality.root')

utils.fill_th1f_hist(hCentrality_Rec, df_filtered, 'fCentralityFT0C')
utils.fill_th1f_hist(hPtReco_by_0_10, df_reco_0_10_wth_rej, 'fPtLnn')
utils.fill_th1f_hist(hPtReco_by_10_30, df_reco_10_30_wth_rej, 'fPtLnn')
utils.fill_th1f_hist(hPtReco_by_30_50, df_reco_30_50_wth_rej, 'fPtLnn')

utils.fill_th1f_hist(hPtReco_by_0_10_dfReco, df_reco_0_10, "fPtLnn")
utils.fill_th1f_hist(hPtReco_by_10_30_dfReco, df_reco_10_30, "fPtLnn")
utils.fill_th1f_hist(hPtReco_by_30_50_dfReco, df_reco_30_50, "fPtLnn")

utils.fill_th1f_hist(hPtReco, df_filtered, 'fPtLnn')
utils.fill_th1f_hist(hPtReco_1, df_filtered, 'fPtLnn')
utils.fill_th1f_hist(hPtRecoIsMatter, df_filtered, 'fPtLnn')
utils.fill_th1f_hist(hPtRecoIsAntiMatter, df_filtered, 'fPtLnn')

utils.fill_th1f_hist(hMassLnn, df_filtered, 'fLnnM')

utils.fill_th1f_hist(hPtReco3H, df_filtered, 'fPt3H')

utils.fill_th1f_hist(hCtReco, df_filtered, 'fCt')

utils.fill_th1f_hist(hDecayLen, df_filtered, 'fDecLen')

utils.fill_th1f_bool_hist(hIsMatter, df_filtered, 'fIsMatter')

utils.fill_th1f_hist(hDecayLen, df_filtered, 'fDecLen')

utils.fill_th2f_hist(h2CentVsPtReco, df_filtered,'fCentralityFT0C', 'fPtLnn')
utils.fill_th2f_hist(h2CentVsPtGen, IsGen_df, 'fCentralityFT0C', 'fAbsGenPt')

utils.number_of_lnn_by_pT(df_reco_0_10, 'fIsReco', 'fPtLnn', 'fSurvivedEventSelection', 'number_reco_lnn_0_10.root')
utils.number_of_lnn_by_pT(df_reco_10_30, 'fIsReco', 'fPtLnn', 'fSurvivedEventSelection', 'number_reco_lnn_10_30.root')
utils.number_of_lnn_by_pT(df_reco_30_50, 'fIsReco', 'fPtLnn', 'fSurvivedEventSelection', 'number_reco_lnn_30_60.root')

utils.fill_th1f_hist(hnTPCClus3H, df, "fNTPCclus3H")
utils.fill_th1f_hist(hCosPA, df, 'fCosPA')

# utils.fill_th1f_hist(hNSigma3H, df, 'fNSigma3H_MC')

hEfficiencyAntiMatter = utils.efficiency(hPtGen, hPtRecoIsAntiMatter, "hEffciencyAntiMatter")
hEfficiencyMatter = utils.efficiency(hPtGen, hPtRecoIsMatter, 'hEfficiencyMatter')
hEfficiency_1 = utils.eff(hPtGen_1, hPtReco_1, "Efficiency_Reweight")

hEfficiency3H = utils.efficiency(hPtGen3H, hPtReco3H, 'hEfficiencyPt3H')
hEfficiencyCt = utils.Ctefficiency(hCtGen, hCtReco, 'hEfficiencyCt')

####### Eff Reco only selections
hEfficiency = utils.eff(hPtGen, hPtReco, "Efficiency")
####### Eff Reco 3H BW
hEfficiency_1 = utils.eff(hPtGen_1, hPtReco_1, "Efficiency_Reweight_3H_curve")
####### Eff Reco Sel + Cent 
hEff_Sel_Cent_0_10 = utils.eff(hPtGen_by_0_10_only_cent, hPtReco_by_0_10, "Efficiency_Sel_Cent_0_10")
hEff_Sel_Cent_10_30 = utils.eff(hPtGen_by_10_30_only_cent, hPtReco_by_10_30, "Efficiency_Sel_Cent_10_30")
hEff_Sel_Cent_30_50 = utils.eff(hPtGen_by_30_50_only_cent, hPtReco_by_30_50, "Efficiency_Sel_Cent_30_50")
####### Eff Reco Sel + Cent + Rej sample
h_eff_0_10 = utils.eff(hPtGen_by_0_10, hPtReco_by_0_10_dfReco, "hEffRW_0_10")
h_eff_10_30 = utils.eff(hPtGen_by_10_30, hPtReco_by_10_30_dfReco, "hEffRW_10_30")
h_eff_30_50 = utils.eff(hPtGen_by_30_50, hPtReco_by_30_50_dfReco, "hEffRW_30_50")

# heff_bf_0_10 = utils.eff(hPtGen, hPtReco_by_0_10, "heff_bf_0_10")
# heff_bf_10_30 = utils.eff(hPtGen, hPtReco_by_10_30, "heff_bf_10_30")
# heff_bf_30_50 = utils.eff(hPtGen, hPtReco_by_30_50, "heff_bf_30_50")

# utils.number_of_generated_reweight_by_centrality_0_10(df_accepted_0_10, df_accepted_10_30, df_accepted_30_50, 'fCentralityFT0C', 'fAbsGenPt')

#Eff_by_Centrality = utils.eff_by_centrality(df, 'fCentralityFT0C', 'fPtLnn', 'fAbsGenPt', 'fIsMatter')

hCtGen.GetXaxis().SetTitle("#it{c#tau}_{Gen} (cm)")
hCtGen.GetYaxis().SetTitle("Counts")

hPtGen.GetXaxis().SetTitle("Gen #it{p}_{T} (GeV/#it{c})")
hPtGen.GetYaxis().SetTitle("Counts")

hMassLnn.GetXaxis().SetTitle("#it{M}_{3H#pi} [GeV/c^{2}]")
hMassLnn.GetYaxis().SetTitle("Counts")

hPtGen_1.GetXaxis().SetTitle("Gen #it{p}_{T} (GeV/#it{c})")
hPtGen_1.GetYaxis().SetTitle("Counts")

output_file = ROOT.TFile(f"{output_file_name}.root", "RECREATE")

hCentrality.Write()
hCentrality_Rec.Write()

hCtReco.Write()
hCtGen.Write()
hPtGen_1.Write()
hMassLnn.Write()

hEfficiency3H.Write()
hEfficiencyCt.Write()

hGenPz.Write()
hGenP.Write()
hGenDecLen.Write()
hDecayLen.Write()

hNSigma3H.Write()

hPtReco.Write()
hPtReco_1.Write()
hPtReco_by_0_10.Write()
hPtReco_by_10_30.Write()
hPtReco_by_30_50.Write()
hPtReco_by_0_10_dfReco.Write()
hPtReco_by_10_30_dfReco.Write()
hPtReco_by_30_50_dfReco.Write()

hPtGen.Write()
hPtGen_1.Write()
hPtGen_by_0_10_only_cent.Write()
hPtGen_by_10_30_only_cent.Write()
hPtGen_by_30_50_only_cent.Write()
hPtGen_by_0_10.Write()
hPtGen_by_10_30.Write()
hPtGen_by_30_50.Write()

hEfficiency.Write()
hEfficiency_1.Write()

hEff_Sel_Cent_0_10.Write()
hEff_Sel_Cent_10_30.Write()
hEff_Sel_Cent_30_50.Write()

h_eff_0_10.Write()
h_eff_10_30.Write()
h_eff_30_50.Write()

h2CentVsPtReco.Write()
h2CentVsPtGen.Write()

hNSigma3H.Write()
hnTPCClus3H.Write()
hCosPA.Write()

hIsReco.Write()
hIsMatter.Write()
hIsSignal.Write()
hIsSuvEvSel8.Write()

output_file.Close()



print(df.loc[(df['fCentralityFT0C'] < 10) & (df['fIsReco'] == False)].shape[0])