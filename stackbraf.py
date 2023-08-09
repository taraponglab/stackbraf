
from joblib import load
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from padelpy import padeldescriptor, from_smiles
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from glob import glob
import os
# import zipfile

def execute_algorithm(smile,name):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    print('   ')
    print('Welcome to StackBRAF model, you can predict the BRAF inhibitory activity of your chemical compound based on the SMILE string')
    print('   ')

    xml_files = glob(base_directory + "/" + "*.xml")
    xml_files.sort()
    FP_list = [
    'AtomPairs2DCount',
    'AtomPairs2D',
    'EState',
    'CDKextended',
    'CDK',
    'CDKgraphonly',
    'KlekotaRothCount',
    'KlekotaRoth',
    'MACCS',
    'PubChem',
    'SubstructureCount',
    'Substructure']
    fp = dict(zip(FP_list, xml_files))

    df = {name : smile}
    df = pd.DataFrame.from_dict(df, orient='index', columns=['Smile'])
    df.index.name='Name'
    smile_path = os.path.join(base_directory, 'stackbraf-prediction/smiles/smile.smi')
    df.to_csv(smile_path, sep='\t', index=False, header=False)

    print(' Task 1: SMILE loading completed')

    target_path = os.path.join(base_directory, 'stackbraf-prediction/fingerprints/')
    for i in FP_list:
        fingerprint = i

        fingerprint_output_file = target_path + ''.join([fingerprint,'_BRAF.csv'])
        fingerprint_descriptortypes = fp[fingerprint]
        padeldescriptor(mol_dir=smile_path,
                        d_file=fingerprint_output_file,
                        descriptortypes= fingerprint_descriptortypes,
                        retainorder=True, 
                        removesalt=True,
                        threads=2,
                        detectaromaticity=True,
                        standardizetautomers=True,
                        standardizenitro=True,
                        fingerprints=True
                        )
    print(' Task 2: Fingerprint calculation completed')

    #individual compounds
    target_path = os.path.join(base_directory, 'ad-analysis/nonoutlier.csv')
    nonoutlier = pd.read_csv(target_path,index_col='LigandID')
    outlier_smile =list(nonoutlier['name'])
    similarity = []
    similarity_max = []
    query = Chem.MolFromSmiles(smile)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(query, 3, nBits=2048)
    for i in outlier_smile:
        mol = Chem.MolFromSmiles(i)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        s = DataStructs.TanimotoSimilarity(fp1,fp2)
        similarity.append(s)
    similarity_max.append(max(similarity))
    similarity_score = pd.DataFrame({'Similarity_nonoutliers': similarity_max}, index=df.index)


    print(' Task 3: AD calculation completed')

    target_path = os.path.join(base_directory, 'stackbraf-prediction/fingerprints/')

    fp = {}
    for i in FP_list:
        fp[i] = pd.read_csv(target_path + ''.join([i,'_BRAF.csv'])).set_index(df.index)
        fp[i] = fp[i].drop('Name', axis=1)
        fp[i].to_csv(target_path + '/fp_'+''.join([i,'.csv']))
    fp_load = [file for file in sorted(glob(os.path.join(target_path, 'fp_*.csv')))]
    #list
    fp_lists = [
    'AtomPairs2D',
    'AtomPairs2DCount',
    'CDK',
    'CDKextended',
    'CDKgraphonly',
    'EState',
    'KlekotaRoth',
    'KlekotaRothCount',
    'MACCS',
    'PubChem',
    'Substructure',
    'SubstructureCount']
    fp_smile_lists = dict(zip(fp_lists, fp_load))
    fp_smile_lists
    fp_smile = {}
    for i in fp_lists:
        fp_smile[i] = pd.read_csv(fp_smile_lists[i],index_col='Name')
    #prediction
    Model = {}
    y_fda_predict = {}
    name = 'XGB'

    target_path = os.path.join(base_directory, 'models/models-fp/')

    for i in fp_lists:
        Model[i] = load(target_path + name + '_reg_' + i + '.joblib')
        y_fda_predict[i] = Model[i].predict(fp_smile[i])

    list
    columns_list = [
    name+'_AtomPairs2D',
    name+'_AtomPairs2DCount',
    name+'_CDK',
    name+'_CDKextended',
    name+'_CDKgraphonly',
    name+'_EState',
    name+'_KlekotaRoth',
    name+'_KlekotaRothCount',
    name+'_MACCS',
    name+'_PubChem',
    name+'_Substructure',
    name+'_SubstructureCount',
    ]

    #save data for next training
    df_predict_xgb=pd.DataFrame.from_dict(y_fda_predict,orient='index').transpose().set_index(df.index)
    df_predict_xgb.columns=columns_list


    print(' Task 4: XGB calculation completed')


    # with zipfile.ZipFile("models/models-fp/SVR_reg_KlekotaRoth.zip","r") as zip_ref:
    #     zip_ref.extractall("models/models-fp/")

    #prediction
    Model = {}
    y_fda_predict = {}
    name = 'SVR'

    for i in fp_lists:
        Model[i] = load(target_path + name + '_reg_' + i + '.joblib')
        y_fda_predict[i] = Model[i].predict(fp_smile[i])

    list
    columns_list = [
    name+'_AtomPairs2D',
    name+'_AtomPairs2DCount',
    name+'_CDK',
    name+'_CDKextended',
    name+'_CDKgraphonly',
    name+'_EState',
    name+'_KlekotaRoth',
    name+'_KlekotaRothCount',
    name+'_MACCS',
    name+'_PubChem',
    name+'_Substructure',
    name+'_SubstructureCount',
    ]

    #save data for next training
    df_predict_svr=pd.DataFrame.from_dict(y_fda_predict,orient='index').transpose().set_index(df.index)
    df_predict_svr.columns=columns_list

    print(' Task 5: SVR calculation completed')

    #prediction
    Model = {}
    y_fda_predict = {}
    name = 'MLP'

    for i in fp_lists:
        Model[i] = load(target_path + name + '_reg_' + i + '.joblib')
        y_fda_predict[i] = Model[i].predict(fp_smile[i])

    list
    columns_list = [
    name+'_AtomPairs2D',
    name+'_AtomPairs2DCount',
    name+'_CDK',
    name+'_CDKextended',
    name+'_CDKgraphonly',
    name+'_EState',
    name+'_KlekotaRoth',
    name+'_KlekotaRothCount',
    name+'_MACCS',
    name+'_PubChem',
    name+'_Substructure',
    name+'_SubstructureCount',
    ]

    #save data for next training
    df_predict_mlp=pd.DataFrame.from_dict(y_fda_predict,orient='index').transpose().set_index(df.index)
    df_predict_mlp.columns=columns_list

    print(' Task 6: MLP calculation completed')


    fp_pf = pd.concat([df_predict_mlp,df_predict_svr,df_predict_xgb], axis=1)

    target_path = os.path.join(base_directory, 'models/StackMLP-SVR-XGB.joblib')
    #import model
    Model = load(target_path)
    #predict class
    res = pd.DataFrame(Model.predict(fp_pf), columns=['pIC50'], index=fp_pf.index)
    res = pd.concat([res, similarity_score], axis=1)

    print(' Task 7: Stack calculation completed')
    print(' Task 8: Prediction result')
    print('   ')
    print('   ')
    print(res)
    print('   ')
    print('   ')
    print('Thank you for using our model.')
    print('   ')
    print('All right reserved 2023')
    print('Dr. Tarapong Srisongkram and Nur Fadhilah Syahid')
    print('   ')

    result = res.to_json(orient="index")

    return result

#execute_algorithm("CC(C)N1C=C(C(=N1)C2=C(C(=CC(=C2)Cl)NS(=O)(=O)C)F)C3=NC(=NC=C3)NCC(C)NC(=O)OC","encorafenib")
