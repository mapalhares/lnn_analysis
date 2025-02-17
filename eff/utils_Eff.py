import ROOT
import pandas as pd 
import numpy as np
import cmath as math
def fill_th1f_hist(h, df, key): 
    for entry in df[key]:
        h.Fill((entry))

def fill_th2f_hist(h, df, key1, key2):
    for entry_1, entry_2 in zip(df[key1], df[key2]):
        h.Fill(entry_1, entry_2)

def fill_th1f_bool_hist(h, df, key):
    for entry in df[key]:
        h.Fill(int(entry))
        
def convert_sel_to_string(selection):
    sel_string = ''
    conj = ' and '
    for _, val in selection.items():
        sel_string = sel_string + val + conj
    return sel_string[:-len(conj)]
        
### reweight a distribution with rejection sampling
def reweight_pt_spectrum(df, var, distribution):
    rej_flag = np.ones(len(df))
    random_arr = np.random.rand(len(df))
    max_bw = distribution.GetMaximum()

    for ind, (val, rand) in enumerate(zip(df[var],random_arr)):
        frac = distribution.Eval(val)/max_bw
        if rand > frac:
            rej_flag[ind] = -1
    ## check if it is a pandas dataframe
    if isinstance(df, pd.DataFrame):
        df[f'rej'] = rej_flag
        return
    df._full_data_frame[f'rej'] = rej_flag
    
def reweight_pt_spectrum_0_10(df, var, distribution):
    rej_flag = np.ones(len(df))
    random_arr = np.random.rand(len(df))
    max_bw = distribution.GetMaximum()

    for ind, (val, rand) in enumerate(zip(df[var],random_arr)):
        frac = distribution.Eval(val)/max_bw
        if rand > frac:
            rej_flag[ind] = -1
    ## check if it is a pandas dataframe
    if isinstance(df, pd.DataFrame):
        df[f'rej_0_10'] = rej_flag
        return
    df._full_data_frame[f'rej_0_10'] = rej_flag

def reweight_pt_spectrum_10_30(df, var, distribution):
    rej_flag = np.ones(len(df))
    random_arr = np.random.rand(len(df))
    max_bw = distribution.GetMaximum()

    for ind, (val, rand) in enumerate(zip(df[var],random_arr)):
        frac = distribution.Eval(val)/max_bw
        if rand > frac:
            rej_flag[ind] = -1
    ## check if it is a pandas dataframe
    if isinstance(df, pd.DataFrame):
        df[f'rej_10_30'] = rej_flag
        return
    df._full_data_frame[f'rej_10_30'] = rej_flag  
    
def reweight_pt_spectrum_30_50(df, var, distribution):
    rej_flag = np.ones(len(df))
    random_arr = np.random.rand(len(df))
    max_bw = distribution.GetMaximum()

    for ind, (val, rand) in enumerate(zip(df[var],random_arr)):
        frac = distribution.Eval(val)/max_bw
        if rand > frac:
            rej_flag[ind] = -1
    ## check if it is a pandas dataframe
    if isinstance(df, pd.DataFrame):
        df[f'rej_30_50'] = rej_flag
        return
    df._full_data_frame[f'rej_30_50'] = rej_flag    
    
        
def h3BB(rigidity, mass):

    p1 = -405.2308
    p2 = 6.169
    p3 = 186.0143
    p4 = -0.3742
    p5 = 5.6339

    betagamma = rigidity / mass
    beta = betagamma / np.sqrt(1 + betagamma**2)
    aa = beta**p4
    bb = np.log(p3 + (1 / betagamma)**p5)
    return (p2 - aa - bb) * p1 / aa

def computeNSigma3H(df, key):
    hNsigma3H = ROOT.TH1F("hNsigma3H", "hNsigma3H", 10, -3, 3)
    hNsigma3H.GetXaxis().SetTitle("n_{#sigma}^{TPC}(^{3}H)")
    hNsigma3H.GetYaxis().SetTitle("Counts")
    canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)

    
    for tpcmom3H in df[key]:
        expBB = h3BB(tpcmom3H, 2.80892113298)
    
        value = (tpcmom3H - expBB) / (0.09*tpcmom3H)
        
        hNsigma3H.Fill(value)
    hNsigma3H.Draw()
    canvas.Draw()
    return hNsigma3H


def apply_pt_rejection(df, pt_shape):
    rej_flag = np.ones(len(df))
    random_arr = np.random.rand(len(df))
    max_bw = pt_shape.GetMaximum()

    for ind, (pt, rand) in enumerate(zip(df['gPt'],random_arr)):
        frac = pt_shape.Eval(pt)/max_bw
        if rand > frac:
            rej_flag[ind] = -1
    df._full_data_frame['rej'] = rej_flag     
        
def efficiency(gen_hist, rec_hist, name):
    eff_hist = gen_hist.Clone(name)
    eff_hist.Reset()
    eff_hist.GetXaxis().SetTitle("#it{p}_{T} [GeV/c^{2}]")
    eff_hist.GetYaxis().SetTitle(r'#epsilon #times Acc')
    eff_hist.GetYaxis().SetRangeUser(0, 0.12)
    eff_hist.GetXaxis().SetRangeUser(2, 8)
    eff_hist.SetTitle(name)
    for iPt in range(1, rec_hist.GetNbinsX() + 1):
        gen_val = gen_hist.GetBinContent(iPt)
        if gen_val < 1e-24:
            continue
        rec_val = rec_hist.GetBinContent(iPt)
        eff_val = rec_val / gen_val
        eff_err = np.sqrt((eff_val * (1 - eff_val) / gen_val))
        # print('iPt: ', iPt, ' eff: ', eff_val, ' +- ', eff_err)
        eff_hist.SetBinContent(iPt, eff_val)
        eff_hist.SetBinError(iPt, eff_err)
    return eff_hist

def Ctefficiency(gen_hist, rec_hist, name):
    eff_hist = gen_hist.Clone(name)
    eff_hist.Reset()
    eff_hist.GetXaxis().SetTitle(r'c#tau (cm)')
    eff_hist.GetYaxis().SetTitle(r'#epsilon #times Acc')
    eff_hist.GetYaxis().SetRangeUser(0, 0.2)
    eff_hist.GetXaxis().SetRangeUser(2.0, 8.0)
    eff_hist.SetTitle(name)
    for iPt in range(1, rec_hist.GetNbinsX() + 1):
        gen_val = gen_hist.GetBinContent(iPt)
        if gen_val < 1e-24:
            continue
        rec_val = rec_hist.GetBinContent(iPt)
        eff_val = rec_val / gen_val
        eff_err = np.sqrt(abs(eff_val * (1 - eff_val) / gen_val))
        # print('iPt: ', iPt, ' eff: ', eff_val, ' +- ', eff_err)
        eff_hist.SetBinContent(iPt, eff_val)
        eff_hist.SetBinError(iPt, eff_err)
    return eff_hist

def eff(hGen, hReco, name):
    # Calculando a eficiência (Reco / Gen), onde hGen e hReco são histogramas
    hEff = hReco.Clone(name)
    hEff.GetXaxis().SetRangeUser(2.0, 8.0)# Clonando o histograma Reco
    hEff.GetXaxis().SetRangeUser(0, 0.01)# Clonando o histograma Reco
    hEff.GetXaxis().SetTitle("#it{p}_{T} [GeV/c^{2}]")
    hEff.GetYaxis().SetTitle(r'#epsilon #times Acc')
    hEff.SetTitle(f"{name}")
    hEff.Divide(hGen)  # Calculando a eficiência dividindo Reco por Gen
    return hEff

def eff_by_centrality(df, key_centrality, key_PtReco, key_PtGen, key_isMatter): 
    output = ROOT.TFile("Efficiency_per_Centrality.root", "RECREATE")
    
    hFT0 = ROOT.TH1F("hFT0", "", 10, 0, 100)
    
    hPtReco_by_0_10 = ROOT.TH1F("hPtReco_by_0_10", "RecoPt - 0-10%", 20, 0., 10.)
    hPtGen_by_0_10 = ROOT.TH1F('hPtGen_by_0_10', "GenPt - 0-10%", 20, 0., 10.)
    
    hPtReco_by_10_30 = ROOT.TH1F("hPtReco_by_10_30", "RecoPt - 10-30%", 20, 0., 10.)
    hPtGen_by_10_30 = ROOT.TH1F('hPtGen_by_10_30', "GenPt - 10-30%", 20, 0., 10.)
    
    hPtReco_by_30_50 = ROOT.TH1F("hPtReco_by_30_50", "RecoPt - 30-50%", 20, 0., 10.)
    hPtGen_by_30_50 = ROOT.TH1F('hPtGen_by_30_50', "GenPt - 30-50%", 20, 0., 10.)
    
    hPtReco_by_50_90 = ROOT.TH1F("hPtReco_by_50_90", "RecoPt - 50-90%", 20, 0., 10.)
    hPtGen_by_50_90 = ROOT.TH1F('hPtGen_by_50_90', "GenPt - 50-90%", 20, 0., 10.)
    
    for entry_1, entry_2, entry_3, entry_4 in zip(df[key_centrality], df[key_PtReco], df[key_PtGen], df[key_isMatter]):
        print(f"Centrality: {entry_1}, PtReco: {entry_2}, PtGen: {entry_3}, isMatter: {entry_4}")
        hFT0.Fill(entry_1)
        
        if 0 <= entry_1 < 10:
            print(f"Preenchendo hPtReco_by_0_10 com {entry_2}, hPtGen_by_0_10 com {entry_3}")
            hPtReco_by_0_10.Fill(entry_2)
            hPtGen_by_0_10.Fill(entry_3)

        elif 10 <= entry_1 < 30:
            print(f"Preenchendo hPtReco_by_10_30 com {entry_2}, hPtGen_by_10_30 com {entry_3}")
            hPtReco_by_10_30.Fill(entry_2)
            hPtGen_by_10_30.Fill(entry_3)

        elif 30 <= entry_1 < 50:
            print(f"Preenchendo hPtReco_by_30_50 com {entry_2}, hPtGen_by_30_50 com {entry_3}")
            hPtReco_by_30_50.Fill(entry_2)
            hPtGen_by_30_50.Fill(entry_3)

        elif 50 <= entry_1 < 100:
            print(f"Preenchendo hPtReco_by_50_90 com {entry_2}, hPtGen_by_50_90 com {entry_3}")
            hPtReco_by_50_90.Fill(entry_2)
            hPtGen_by_50_90.Fill(entry_3)
                    
    h_eff_0_10 = efficiency(hPtGen_by_0_10, hPtReco_by_0_10, "Eff_0_10")
    h_eff_10_30 = efficiency(hPtReco_by_10_30, hPtReco_by_10_30, "Eff_10_30")
    h_eff_30_50 = efficiency(hPtGen_by_30_50, hPtReco_by_30_50, "Eff_30_50")
    h_eff_50_90 = efficiency(hPtGen_by_50_90, hPtReco_by_50_90, "Eff_50_90")
            
    h_eff_0_10.Write()
    h_eff_10_30.Write()
    h_eff_30_50.Write()
    h_eff_50_90.Write()
    
    output.Close()
        
    return output

def correct_and_convert_df(df):

    #lnn momentum and E_lnn
    pi_V = ROOT.TLorentzVector()
    triton_V = ROOT.TLorentzVector()
    lnn_V = ROOT.TLorentzVector()

    PiMinus_Mass = 0.139570 #o2::constants::physics::MassPiMinus
    H3_Mass = 2.808921 #o2::constants::physics::Triton
    LnnMass = 2.9937 #HypHI collaboration
    
    #Lnn lorentz vector
    for index, row in df.iterrows():
        Pi_pT = row['fPtPi']
        Pi_Eta = row['fEtaPi']
        Pi_Phi = row['fPhiPi']
        H3_pT = row['fPt3H']
        H3_Eta = row['fEta3H']
        H3_Phi = row['fPhi3H']
            
        pi_V.SetPtEtaPhiM(Pi_pT, Pi_Eta, Pi_Phi, PiMinus_Mass)
        triton_V.SetPtEtaPhiM(H3_pT, H3_Eta, H3_Phi, H3_Mass)

        lnn_V = pi_V + triton_V

        df.at[index, 'fPtLnn'] = lnn_V.Pt()
        df.at[index, 'fEtaLnn'] = lnn_V.Eta()
        df.at[index, 'fPhiLnn'] = lnn_V.Phi()
        df.at[index, 'fLnnM'] = lnn_V.M()
        df.at[index, 'fLnnRapidity'] = lnn_V.Rapidity() 
        df.at[index, 'fLnnPseudoRapidity'] = lnn_V.PseudoRapidity()       
            
    #MC
    df.eval('fPxLnn = fPtLnn *cos(fPhiLnn)', inplace = True) #Px_Lnn
    df.eval('fPyLnn = fPtLnn *sin(fPhiLnn)', inplace = True) #Py_Lnn
    df.eval('fPzLnn = fPtLnn *sinh(fEtaLnn)', inplace = True) #Pz_Lnn
    df.eval('fPLnn = fPtLnn *cosh(fEtaLnn)', inplace = True) #P total

     #MC
    df.eval('fGenPz = fGenPt * sinh(fGenEta)', inplace=True) #Coordinate z to momentum generated
    df.eval('fGenP = sqrt(fGenPt**2 + fGenPz**2)', inplace=True) #Momentum generated total
    df.eval("fAbsGenPt = abs(fGenPt)", inplace=True) #Absolute values to GenPt
                
    #Variables of interest
    df.eval('fDecLen = sqrt(fXDecVtx**2 + fYDecVtx**2 + fZDecVtx**2)', inplace = True) #Lnn trajectory lenght [DecVtx = PCACandidateVtx - primVtx]
    df.eval('fCosPA = (fPxLnn*fXDecVtx + fPyLnn*fYDecVtx + fPzLnn*fZDecVtx) / (fPLnn*fDecLen)', inplace=True) #Lnn_CosPA
    df.eval('fGenDecLen = sqrt(fGenXDecVtx**2 + fGenYDecVtx**2 + fGenZDecVtx**2)', inplace=True) #Lnn decay lenght generated

    df.eval('fCt = (fDecLen * 2.9937)/fPLnn', inplace=True) #Proper ct Lnn data
    df.eval('fGenCt = (2.9937 * fGenDecLen)/ fGenP', inplace = True) #Proper ct Lnn MC
    df.eval('fMassTOF = sqrt(fMassTrTOF)', inplace = True) 

    # Dataframe with cuts
    
    IsGen_df = df.loc[(df['fIsSignal'] == True) & (df['fLnnRapidity'] < 0.5), ['fSurvivedEventSelection','fIsSignal','fAbsGenPt', 'fGenCt', 'fGenPt3H', 'fGenDecLen', 'fGenPz', 'fGenP', 'fCentralityFT0C', 'fIsReco']]
    # IsReco_df_Matter = df.loc[(df['fIsReco'] == True) & (df['fIsMatter'] == True), ['fPt3H', 'fPtLnn', 'fCt', 'fLnnM', 'fCosPA', 'fNSigma3H']]
    # IsReco_df_AntiMatter = df.loc[(df['fIsReco'] == True) & (df['fIsMatter'] == False), ['fPt3H', 'fPtLnn', 'fCt', 'fLnnM', 'fCosPA']]
    
    
    return df, IsGen_df

# def number_of_generated_and_reco_by_centrality_0_10(df, df_gen, cent, gent_pT, reco_pT): 
#     output_file = ROOT.TFile("Number_of_generated_and_reco_lnn__per_centrality.root", "RECREATE")
#     number_of_gen_0_10 = ROOT.TH1F("number_of_gen_0_10", "number_of_gen_0_10", 6, 2, 8)
#     number_of_gen_10_30 = ROOT.TH1F("number_of_gen_0_10", "number_of_gen_0_10", 6, 2, 8)
#     number_of_gen_30_50 = ROOT.TH1F("number_of_gen_0_10", "number_of_gen_0_10", 6, 2, 8)
    
#     number_of_reco_0_10 = ROOT.TH1F("number_of_reco_0_10", "number_of_reco_0_10", 6, 2, 8)
#     number_of_reco_10_30 = ROOT.TH1F("number_of_reco_10_30", "number_of_reco_10_30", 6, 2, 8)
#     number_of_reco_30_50 = ROOT.TH1F("number_of_reco_30_50", "number_of_reco_30_50", 6, 2, 8)
    
#     for centrality, gen_pT in (df_gen[cent], df_gen[gent_pT]):
#             if centrality >= 0 and centrality < 10:
#                 number_of_gen_0_10.Fill(gen_pT)
#             elif centrality >= 10 and centrality < 30:
#                 number_of_gen_10_30.Fill(gen_pT)
#             elif centrality >= 30 and centrality < 50:
#                 number_of_gen_30_50.Fill(gen_pT)
            
#     for centrality, reco_pT in zip(df[cent], df[reco_pT]): 
#         if centrality >= 0 and centrality < 10:
#             number_of_reco_0_10.Fill(reco_pT)
#         elif centrality >= 10 and centrality < 30:
#             number_of_reco_10_30.Fill(reco_pT)
#         elif centrality >= 30 and centrality < 50:
#             number_of_reco_30_50.Fill(reco_pT)
            
#     number_of_gen_0_10.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_gen_0_10.GetYaxis().SetTitle("Number of candidates")
#     number_of_gen_0_10.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_gen_0_10.GetYaxis().SetTitle("Number of candidates")
#     number_of_gen_30_50.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_gen_30_50.GetYaxis().SetTitle("Number of candidates")
    
#     number_of_reco_0_10.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_reco_0_10.GetYaxis().SetTitle("Number of candidates")
#     number_of_reco_10_30.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_reco_10_30.GetYaxis().SetTitle("Number of candidates")
#     number_of_reco_30_50.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_reco_30_50.GetYaxis().SetTitle("Number of candidates")
    
#     number_of_gen_0_10.Write()
#     number_of_gen_10_30.Write()
#     number_of_gen_30_50.Write()
#     number_of_reco_0_10.Write()
#     number_of_reco_10_30.Write()
#     number_of_reco_30_50.Write()
    
#     output_file.Close()
    
#     return output_file

def number_of_reco_by_centrality(df_gen, cent, isRec, output_filename): 
    output_file = ROOT.TFile(output_filename, "RECREATE")

    # Criando histograma
    histograms = ROOT.TH1F("number_of_reco_by_centrality", "Reconstructed Candidates per Centrality", 3, 0, 3)

    # Definindo rótulos dos bins
    centrality_labels = ["0-10%", "10-30%", "30-50%"]
    for i, label in enumerate(centrality_labels):
        histograms.GetXaxis().SetBinLabel(i + 1, label)

    # Inicializando contadores fora do loop
    i_0_10, i_10_30, i_30_50 = 0, 0, 0  

    # Verificando se 'isRec' está nas colunas do DataFrame
    if isRec not in df_gen.columns:
        raise ValueError(f"Column '{isRec}' not found in DataFrame")

    # Iterando sobre os dados
    for centrality, isR in zip(df_gen[cent], df_gen[isRec]):  
        if isR == True:  # Se foi reconstruído
            if 0 <= centrality <= 10:
                i_0_10 += 1
            elif 10 <= centrality <= 30:
                i_10_30 += 1
            elif 30 <= centrality <= 50:
                i_30_50 += 1

    # Desativar caixa de estatísticas
    ROOT.gStyle.SetOptStat(0)

    # Preenchendo histograma corretamente
    histograms.SetBinContent(1, i_0_10)  # Primeiro bin (0-10%)
    histograms.SetBinContent(2, i_10_30) # Segundo bin (10-30%)
    histograms.SetBinContent(3, i_30_50) # Terceiro bin (30-50%)

    # Configurando títulos dos eixos
    histograms.GetXaxis().SetTitle("Centrality Bins")
    histograms.GetYaxis().SetTitle("Number of Candidates")

    # Escrevendo e salvando no arquivo ROOT
    histograms.Write()
    output_file.Close()
    
    return output_filename

def number_of_reco_by_centrality_RW(df_reco_0_10, df_reco_10_30, df_reco_30_50, isRec, output_filename): 
    output_file = ROOT.TFile(output_filename, "RECREATE")

    # Criando histograma
    histograms = ROOT.TH1F("number_of_reco_by_centrality_RW", "Reconstructed Candidates per Centrality", 3, 0, 3)

    # Definindo rótulos para os bins
    centrality_labels = ["0-10%", "10-30%", "30-50%"]
    for i, label in enumerate(centrality_labels):
        histograms.GetXaxis().SetBinLabel(i + 1, label)

    # Verificando se 'isRec' está nas colunas
    for df in [df_reco_0_10, df_reco_10_30, df_reco_30_50]:
        if isRec not in df.columns:
            raise ValueError(f"Column '{isRec}' not found in DataFrame")

    # Contando eventos reconstruídos por centralidade
    i_0_10 = df_reco_0_10[isRec].sum()
    i_10_30 = df_reco_10_30[isRec].sum()
    i_30_50 = df_reco_30_50[isRec].sum()

    ROOT.gStyle.SetOptStat(0)  # Desativar caixa de estatísticas

    # Preenchendo histograma corretamente
    histograms.SetBinContent(1, i_0_10)  # Primeiro bin (0-10%)
    histograms.SetBinContent(2, i_10_30) # Segundo bin (10-30%)
    histograms.SetBinContent(3, i_30_50) # Terceiro bin (30-50%)

    # Configurando títulos dos eixos
    histograms.GetXaxis().SetTitle("Centrality Bins")
    histograms.GetYaxis().SetTitle("Number of Candidates")

    histograms.Write()
    output_file.Close()
    
    return output_filename

def number_of_gen_by_centrality(df, cent, isSignal, output_filename): 
    output_file = ROOT.TFile(output_filename, "RECREATE")

    # Criando histograma
    histograms = ROOT.TH1F("number_of_gen_by_centrality", "Generated Candidates per Centrality", 3, 0, 3)

    # Definindo rótulos para os bins
    centrality_labels = ["0-10%", "10-30%", "30-50%"]
    for i, label in enumerate(centrality_labels):
        histograms.GetXaxis().SetBinLabel(i + 1, label)

    # Contadores de eventos
    i_gen_0_10, i_gen_10_30, i_gen_30_50 = 0, 0, 0    

    # Iterando sobre os dados
    for centrality, isS in zip(df[cent], df[isSignal]):  
        if isS:  # Apenas candidatos gerados
            if 0 <= centrality <= 10:
                i_gen_0_10 += 1
            elif 10 <= centrality <= 30:
                i_gen_10_30 += 1
            elif 30 <= centrality <= 50:
                i_gen_30_50 += 1

    # Preenchendo histograma corretamente
    histograms.SetBinContent(1, i_gen_0_10)  # Primeiro bin (0-10%)
    histograms.SetBinContent(2, i_gen_10_30) # Segundo bin (10-30%)
    histograms.SetBinContent(3, i_gen_30_50) # Terceiro bin (30-50%)

    # Configurando títulos dos eixos
    histograms.GetXaxis().SetTitle("Centrality Bins")
    histograms.GetYaxis().SetTitle("Number of Candidates")

    histograms.Write()
    output_file.Close()
    
    return output_filename

def number_of_gen_by_centrality_RW(df_gen_0_10, df_gen_10_30, df_gen_30_50, isRec, output_file): 
    output_file_name = ROOT.TFile(output_file, "RECREATE")

    # Criando dicionário de histogramas
    histograms = ROOT.TH1F("number_of_gen_by_centrality_RW", "Generated Candidates per Centrality", 3, 0, 3)

    
    centrality_labels = ["0-10%", "10-30%", "30-50%"]
    for i, label in enumerate(centrality_labels):
        histograms.GetXaxis().SetBinLabel(i + 1, label)

    i_0_10 = df_gen_0_10[(df_gen_0_10[isRec])].shape[0]
    i_10_30 = df_gen_10_30[(df_gen_10_30[isRec])].shape[0]
    i_30_50 = df_gen_30_50[(df_gen_30_50[isRec])].shape[0]

    ROOT.gStyle.SetOptStat(0)  # Desativar caixa de estatísticas

    # Preenchendo histograma com os valores
    histograms.SetBinContent(1, i_0_10)
    histograms.SetBinContent(2, i_10_30)
    histograms.SetBinContent(3, i_30_50)
    
    histograms.GetXaxis().SetTitle("Centrality Bins")
    histograms.GetYaxis().SetTitle("Number of Candidates")

    histograms.Write()

    output_file_name.Close()
    
    return output_file_name

def number_of_lnn_by_pT(df_0_10, isRec, pT_reco, isSurvidedSel8, output_file): 
    output_file_name = ROOT.TFile(output_file, "RECREATE")

    # Criando dicionário de histogramas
    histograms = ROOT.TH1F("number_of_reco_by_pT", "Generated Candidates per Centrality", 20, 0, 10)

    n_reco = 0
    for isR, pT, isSel8 in zip(df_0_10[isRec], df_0_10[pT_reco], df_0_10[isSurvidedSel8]):
        if isSel8:
            if isR:
                histograms.Fill(pT)
            n_reco =+1
    
    # Normalizando pelo número de eventos e largura do bin
    if n_reco > 0:
        histograms.Scale(1.0 / (n_reco * histograms.GetBinWidth(1)))
        
    histograms.GetXaxis().SetTitle("p_{T}")
    histograms.GetYaxis().SetTitle("1/N_{evts} dN/dp_{t}")

    # Salvando histograma no arquivo de saída
    output_file_name.cd()
    histograms.Write()
    output_file_name.Close()

# def number_of_generated_reweight_by_centrality_0_10(df_0_10, df_10_30, df_30_50, cent, gent_pT): 
#     output_file = ROOT.TFile("Number_of_generated_reweight_lnn_centrality.root", "RECREATE")
#     number_of_gen_0_10 = ROOT.TH1F("number_of_gen_0_10", "number_of_gen_0_10", 10, 0, 10)
#     number_of_gen_10_30 = ROOT.TH1F("number_of_gen_0_10", "number_of_gen_0_10", 10, 0, 10)
#     number_of_gen_30_50 = ROOT.TH1F("number_of_gen_0_10", "number_of_gen_0_10", 10, 0, 10)
    
    
#     for centrality, gent_PT in zip(df_0_10[cent],df_0_10[gent_pT]): 
#         if centrality >= 0 and centrality < 10:
#             number_of_gen_0_10.Fill(gent_PT)
            
#     for centrality, gent_PT in zip(df_10_30[cent],df_10_30[gent_pT]): 
#         if centrality >= 0 and centrality < 10:
#             number_of_gen_10_30.Fill(gent_PT)
            
#     for centrality, gent_PT in zip(df_30_50[cent],df_30_50[gent_pT]): 
#         if centrality >= 0 and centrality < 10:
#             number_of_gen_30_50.Fill(gent_PT)            
    
#     number_of_gen_0_10.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_gen_0_10.GetYaxis().SetTitle("Number of candidates")
#     number_of_gen_0_10.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_gen_0_10.GetYaxis().SetTitle("Number of candidates")
#     number_of_gen_30_50.GetXaxis().SetTitle("#it{p}_{T}")
#     number_of_gen_30_50.GetYaxis().SetTitle("Number of candidates")
    
    
#     number_of_gen_0_10.Write()
#     number_of_gen_10_30.Write()
#     number_of_gen_30_50.Write()
    
#     output_file.Close()
    
#     return output_file
    
