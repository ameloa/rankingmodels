# This file contains some utils to help load and play around with choice data
import copy
import pandas as pd
import numpy as np
import torch as t

HOMELANG_TO_PROGRAMS = {'CC-Chinese Cantonese': 'CB', 'SP-Spanish': 'SB'}

def clean_dataframe(df, program_covariates, distances_df):
    data_list = []
    for i, row in df.iterrows():
        student = row['studentno']
        first_round = str(row['first_participated_round'])
        programs = row['r'+first_round+'_ranked_programs']
        ranks = row['r'+first_round+'_listed_ranks']
        ctip = row['ctip1']
        homelang = row['homelang_desc']
        year = row['year']
        num_ranked = len(programs)
        if student in distances_df.index and all([program in program_covariates.index and program in distances_df.columns for program in programs]):
            data_list.append([student, programs, num_ranked, ranks, year, ctip, homelang])
    dataset = pd.DataFrame(data_list, columns = ['student', 'programs', 'num_ranked', 'ranks', 'year', 'ctip', 'homelang'])
    return dataset

def remove_repeats(entry):
    # if someone voted for the" same candidate twice, it will remove the repeats
    # while still preserving the order of the original list
    if len(list(np.array(entry)[np.sort(np.unique(entry, return_index=True)[1])])) == 0:
        print(entry)
    return list(np.array(entry)[np.sort(np.unique(entry, return_index=True)[1])])

def final_choice(entry):
    return [entry[-1]]

def prep_for_pytorch(ballot, n):
    voters, choices, choice_sets, chosen_sets = ballot['student_id'].values, ballot['choices'].values, ballot['choice_sets'].values, ballot['chosen_sets'].values
    choice_sets_concat, choices_concat, chosen_sets_concat, whose_choice_concat = [], [], [], []
    for idx, choice_set in enumerate(choice_sets):
        assert(len(choice_set)==len(chosen_sets[idx])==len(choices[idx]))
        choice_sets_concat += choice_set
        chosen_sets_concat += chosen_sets[idx]
        choices_concat += choices[idx]
        whose_choice_concat += [voters[idx]]*len(choices[idx])
    
    choices = choices_concat
    choice_set_lengths = np.array([len(choice_set) for choice_set in choice_sets_concat])
    chosen_set_lengths = np.array([len(chosen_set) for chosen_set in chosen_sets_concat])
    x_extra = np.stack([np.array(whose_choice_concat), choice_set_lengths, chosen_set_lengths], axis=1)
    slots_chosen = np.array([choice_set.index(choices[idx]) for idx, choice_set in enumerate(choice_sets_concat)])
    
    kmax = choice_set_lengths.max() #should always be size of universe/greater than max chosen set length
    padded_choice_sets = np.full([len(choice_sets_concat), kmax], fill_value=n, dtype=np.compat.long)
    choice_sets = np.concatenate(choice_sets_concat)
    padded_choice_sets[np.arange(kmax)[None, :] < choice_set_lengths[:, None]] = choice_sets

    padded_chosen_sets = np.full([len(chosen_sets_concat), kmax], fill_value=n, dtype=np.compat.long)
    chosen_sets = np.concatenate(chosen_sets_concat)
    padded_chosen_sets[np.arange(kmax)[None, :] < chosen_set_lengths[:, None]] = chosen_sets
    x = np.stack([padded_choice_sets, padded_chosen_sets], axis=-1)

    return list(map(t.from_numpy, [x, x_extra, slots_chosen]))

def prep_valset(data, program_data, prog_codex):
    restricted_program_lookup = {'CB':[], 'SB':[]}
    codex_programs = copy.deepcopy(prog_codex)
    n=len(codex_programs)
    
    data['program_id'] = [[codex_programs.index(program) for program in program_list] 
                          if all([(program in codex_programs) and (program in program_data.index) for program in program_list]) 
                          else None
                          for program_list in data.programs]
    data = data.dropna(subset=['program_id'])
    all_restricted = []
    for idx, prog in enumerate(codex_programs):
        program_type = program_data.loc[prog, 'program_type']
        if program_type in ['CB', 'SB']:
            all_restricted.append(idx)
            restricted_program_lookup[program_type].append(idx)
    
    cat=pd.Categorical(data['student'])
    data['student_id'] = cat.codes
    codex_student = list(cat.categories)
    codex_ctip = np.zeros(len(codex_student))

    def choice_set_func(entry, univ, language):
        eligibility_list = []
        if language in HOMELANG_TO_PROGRAMS.keys():
            eligibility_list = restricted_program_lookup[HOMELANG_TO_PROGRAMS[language]]
        univ.extend(eligibility_list)
        for item in entry:
            if item not in univ:
                univ.append(item)
        choice_sets = []
        choice_sets.append(list(univ))
        for idx, item in enumerate(entry[1:]):
            univ.remove(entry[idx])
            choice_sets.append(list(univ))
        return choice_sets

    def chosen_set_func(entry):
        chosen_sets = [[]]
        for idx, item in enumerate(entry[1:]):
            chosen_set = chosen_sets[-1].copy()
            chosen_set.extend([entry[idx]])
            chosen_sets.append(chosen_set)
        return chosen_sets

    ballots = []
    for year in data.year.unique():
        year_data = data[data["year"] == year]
        subset=list(np.arange(len(codex_programs)))
        for item in all_restricted:
            if item in subset:
                subset.remove(item)

        ballot_sort = []
        for ind, item in enumerate(year_data[['program_id', 'ranks']].values):
            if year_data.iloc[ind]['ctip']:
                codex_ctip[year_data.iloc[ind]['student_id']] = 1
            sort_idx = np.argsort(item[1])
            ballot_sort.append([year_data.iloc[ind]['student_id'], 
                                year_data.iloc[ind]['year'], 
                                year_data.iloc[ind]['homelang'], 
                                list(np.array(item[0])[sort_idx]), 
                                list(np.array(item[1])[sort_idx])])
        ballot = pd.DataFrame(ballot_sort, columns = ['student_id', 
                                                      'year',
                                                      'homelang',
                                                      'program_id', 
                                                      'ranks'])
        indices_to_remove = np.array([~np.all(np.array(item) == (np.arange(len(item))+1)) for item in ballot['ranks'].values])
        ballot = ballot[~indices_to_remove]

        ballot['choices'] = ballot['program_id'].apply(remove_repeats)
        ballot['choice_sets'] = ballot.apply(lambda x: choice_set_func(x['choices'], list(subset), language=x['homelang']), axis=1)
        ballot['chosen_sets'] = ballot['program_id'].apply(lambda x: chosen_set_func(x))
        ballots.append(ballot)
    ballot = pd.concat(ballots, ignore_index=True)
    ds = prep_for_pytorch(ballot, n)
    return ds, codex_student, codex_ctip.astype(bool)

def prep_dataset(data, program_data):
    restricted_program_lookup = {'CB':[], 'SB':[]}
    cat=pd.Categorical(data.programs.explode())
    codex_programs = list(cat.categories)
    data['program_id'] = [[codex_programs.index(program) for program in program_list] for program_list in data.programs]
    n=len(codex_programs)

    all_restricted = []
    for idx, prog in enumerate(codex_programs):
        program_type = program_data.loc[prog, 'program_type']
        if program_type in ['CB', 'SB']:
            all_restricted.append(idx)
            restricted_program_lookup[program_type].append(idx)
    
    cat=pd.Categorical(data['student'])
    data['student_id'] = cat.codes
    codex_student = list(cat.categories)
    codex_ctip = np.zeros(len(codex_student))
    
    def choice_set_func(entry, univ, language):
        eligibility_list = []
        if language in HOMELANG_TO_PROGRAMS.keys():
            eligibility_list = restricted_program_lookup[HOMELANG_TO_PROGRAMS[language]]
        univ.extend(eligibility_list)
        for item in entry:
            if item not in univ:
                univ.append(item)
        choice_sets = []
        choice_sets.append(list(univ))
        for idx, item in enumerate(entry[1:]):
            univ.remove(entry[idx])
            choice_sets.append(list(univ))
        return choice_sets

    def chosen_set_func(entry):
        chosen_sets = [[]]
        for idx, item in enumerate(entry[1:]):
            chosen_set = chosen_sets[-1].copy()
            chosen_set.extend([entry[idx]])
            chosen_sets.append(chosen_set)
        return chosen_sets
    
    ballots = []
    for year in data.year.unique():
        year_data = data[data["year"] == year]
        subset = list(year_data.program_id.explode().unique())
        for item in all_restricted:
            if item in subset:
                subset.remove(item)
        ballot_sort = []
        for ind, item in enumerate(year_data[['program_id', 'ranks']].values):
            if year_data.iloc[ind]['ctip']:
                codex_ctip[year_data.iloc[ind]['student_id']] = 1
            sort_idx = np.argsort(item[1])
            ballot_sort.append([year_data.iloc[ind]['student_id'], 
                                year_data.iloc[ind]['year'], 
                                year_data.iloc[ind]['homelang'], 
                                list(np.array(item[0])[sort_idx]), 
                                list(np.array(item[1])[sort_idx])])
        ballot = pd.DataFrame(ballot_sort, columns = ['student_id', 
                                                      'year',
                                                      'homelang',
                                                      'program_id', 
                                                      'ranks'])
        indices_to_remove = np.array([~np.all(np.array(item) == (np.arange(len(item))+1)) for item in ballot['ranks'].values])
        ballot = ballot[~indices_to_remove]

        ballot['choices'] = ballot['program_id'].apply(remove_repeats)
        ballot['choice_sets'] = ballot.apply(lambda x: choice_set_func(x['choices'], list(subset), language=x['homelang']), axis=1)
        ballot['chosen_sets'] = ballot['choices'].apply(lambda x: chosen_set_func(x))
        ballots.append(ballot)
    ballot = pd.concat(ballots, ignore_index=True)
    ds = prep_for_pytorch(ballot, n)

    return ds, codex_student, codex_programs, codex_ctip.astype(bool), ballot