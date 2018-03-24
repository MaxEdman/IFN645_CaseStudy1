def preprocess():
    # Import pandas
    import pandas as pd
    df = pd.read_csv("datasets/organics.csv")
    
    # Drops irrelevant columns.
    df = df.drop(['CUSTID', 'LCDATE', 'ORGANICS', 'AGEGRP1', 'AGEGRP2', 'NEIGHBORHOOD'], axis = 1)
    
    # Calculates the years between DOB and EDATE and adds that value to age for missing values.
    from datetime import datetime
    import numpy as np
    dateformat = '%Y-%m-%d'
    edate = pd.Timestamp(df['EDATE'][0])
    df['DOB'] = pd.to_datetime(df['DOB'], format=dateformat)    # 1
    df['DOB'] = df['DOB'].where(df['DOB'] < edate, df['DOB'] -  np.timedelta64(100, 'Y'))   # 2
    df['AGE'] = (edate - df['DOB']).astype('<m8[Y]')    # 3
    # Drops Edate and Dob since the Age col is the only one needed.
    df = df.drop(['EDATE', 'DOB'], axis = 1)
    
    # denote errorneous values in AFFL column. Should be on scale 1-30.
    mask = df['AFFL'] < 1
    df.loc[mask, 'AFFL'] = 1
    mask = df['AFFL'] > 30
    df.loc[mask, 'AFFL'] = 30

    # Fill mean values for AFFL column.
    df['AFFL'].fillna(df['AFFL'].mean(), inplace=True)
    # Convert the scale to integer. Not sure if this is necessary.
    df['AFFL'] = df['AFFL'].astype(int)
    
    # Fills mean values based on age for loyalty time. 
    means = df.groupby(['AGE'])['LTIME'].mean()
    df = df.set_index(['AGE'])
    df['LTIME'] = df['LTIME'].fillna(means)
    df = df.reset_index()
    
    # Fills all unknown values of gender with U, which is the same thing basically.
    df['GENDER'].fillna('U', inplace=True)
    
    # One-hot-encodes columns with REGION, TV_REGION and NGROUP.
    df = pd.get_dummies(df)
    
    # Returns the dataframe.
    return df