def GetMissingValueDetails(train, bIDandUniqueColInvestigation = False, bRowMissingInvestigation = False):
    nTotalRowsCount = len(train)
    # Get count of missing variables
    listMissingCount = []
    for columnName in train.columns:  # columnName = "CAT1"
        listMissingCount.append(round(train[columnName].isnull().sum(), 0))

    col_class_factor_len = []
    for columnName in train.columns:  # columnName = "CAT1"
        if hasattr(train[columnName], 'cat'): # train[columnName].dtypes == r"category"
            col_class_factor_len.append(len(train[columnName].unique()))
        else:
            col_class_factor_len.append(0)

    #Make DF with these data # , round((listMissingCount/len(train))*100,0)
    Df = pd.DataFrame(train.dtypes).reset_index()
    Df.columns = ["Features", "Class"]
    Df['MissingCount'] = pd.Series(listMissingCount, index=Df.index)
    Df['MissingCountPercentage'] = pd.Series((pd.to_numeric(Df["MissingCount"], errors = "coerce")/len(Df))*100, index=Df.index)
    Df['FactorLength'] = pd.Series(col_class_factor_len, index=Df.index)

    #How many fields are having missing value
    missingFieldValueCount = len(Df.loc[Df["MissingCount"] > 0])  # df.loc[df['column_name'].isin(some_values)]

    #Overall what is % of data having at least one missing fields
    percentageMissingFields = (1 - (len(train.dropna())/nTotalRowsCount)) * 100

    if bIDandUniqueColInvestigation == True:
    #ID Column - Having each value unique. Applicable for Char, Integer and Factor only. But not necessarily all need to be removed by Analyst
    # Single Unique Value column. Applicable for any kind of column
        col_IDValueColumn = []
        col_SingleUniqueValueColumn = []
        for i in range(0, len(col_class_factor_len), 1):
            if col_class_factor_len[i] == 1:
                col_SingleUniqueValueColumn.append(train.columns[i])
            elif col_class_factor_len[i] == nTotalRowsCount:
                col_IDValueColumn.append(train.columns[i])
            else:
                col_SingleUniqueValueColumn  = None
                col_IDValueColumn = None
    # bIDandUniqueColInvestigation

    missingValue = {'DfMissingDetails': Df, 'nMissingFieldValueCount': missingFieldValueCount, 'fPercentageMissingFields': percentageMissingFields}
#, 'col_SingleUniqueValueColumn': col_SingleUniqueValueColumn, 'col_IDValueColumn': col_IDValueColumn}  # , DfRow = DfRow

    return missingValue
 #return {'Df': Df, 'missingFieldValueCount': missingFieldValueCount, 'percentageMissingFields': percentageMissingFields}
# getMissingValueDetails

def GetDataDictionary(train, nTopNLavels = 5):
    # Get top N of factor variables
    listFacFeatures = []
    for columnName in train.columns:  # columnName = "CAT1"
        if hasattr(train[columnName], 'cat'): # train[columnName].dtypes == r"category"
            listFacFeatures.append(columnName)

    dtResult = []
    for facFeature in listFacFeatures:  # [75:78] [0:2] facFeature = listFacFeatures[1]
        df = pd.DataFrame(train.groupby(facFeature).size()).reset_index()
        df.columns = ["Feature", "Count"]
        df.sort_values("Count", ascending=False, inplace=True)
        dtResult.append(tuple((facFeature, str(tuple(df.iloc[0:min(len(df), nTopNLavels),0].tolist())))))

    # Get spread of numeric variables
    listNumericColumns = list(train.select_dtypes(include=[np.number]).columns.values)
    for numColumn in listNumericColumns:  # numColumn = listNumericColumns[0]
        minValue = round(train[numColumn].min(), 2)
        maxValue = round(train[numColumn].max(), 2)
        meanValue = round(train[numColumn].mean(), 2)
        skewValue = round(train[numColumn].skew(), 2)
        TopValues = "Min:", minValue, ", Max:", maxValue, ", Mean:", meanValue, ", Skew:", skewValue
        dtResult.append(tuple((numColumn, TopValues)))

    df = pd.DataFrame(dtResult, columns =["Features", "TopValues"])

    mv = GetMissingValueDetails(train, bIDandUniqueColInvestigation = True)
    mv['DfMissingDetails']

    dataDic = df.join(mv['DfMissingDetails'].set_index(['Features']),on=['Features'], how='inner')

    return dataDic
    # GetDataDictionary

# Get Ctaegorical group for mean of respective numeric variable. Count and respective % of any vector colCat = "CAT114" colNum = "CONT1" bConvertToLower = True
def GetGroup_Cat_Num(train, colCat,colNum, iRoundCutoffDigit = 2, bConvertToLower = False, nSelectTopN = 10):
    # Select only required column
    df = pd.DataFrame(train[[colCat, colNum]])

    # Convert to lower if required
    if bConvertToLower:
        df[colCat] = df[colCat].str.lower()

    # Take mean of Numeric variable grouped by category variable. Sort Desecnding
    df = df.groupby(colCat).mean().reset_index()
    df = df.sort_values(colNum, ascending = False)

    # Trim the trainling digit
    df[colNum] = df[colNum].round(iRoundCutoffDigit)

    # Take top N
    df = df.head(min(len(df), nSelectTopN))

    return(df)  # GetGroup_Cat_Num

# Get Categorical group for categorical variable. Count and respective % of any vector
def GetGroup_Cat(train, colCat, iRoundCutoffDigit = 2, bConvertToLower = False, nSelectTopN = 10):
    # Select only required column
    df = pd.DataFrame(train[[colCat]])  # colCat= "CAT114"

    # Convert to lower if required
    if bConvertToLower:
        df[colCat] = df[colCat].str.lower()

    # Take mean of Numeric variable grouped by category variable. Sort Desecnding
    df = df.groupby(colCat).size().reset_index()
    df.columns = [colCat,"Count"]

    df = df.sort_values("Count", ascending = False)

    tempSum = df["Count"].sum()
    df["CountPercentage"] = df["Count"]/tempSum

    # Trim the trainling digit
    df["CountPercentage"] = df["CountPercentage"].round(iRoundCutoffDigit)

    # Take top N
    df = df.head(min(len(df), nSelectTopN))

    return(df)  # GetGroup_Cat

# This provides descriptive Analytics for single numeric features
def Desc_Numeric_Single (train, listNumericFeatures = None, strResponse = None, filePdf = "Desc_Numeric_Single.pdf", folderImageDescriptive = "Images/Descriptive/"):

    if listNumericFeatures == None:
        listNumericFeatures = list(train.select_dtypes(include=[np.number]).columns.values)

    pdf = PdfPages(folderImageDescriptive + filePdf)

    for numericColumn in listNumericFeatures:  # numericColumn = listNumericFeatures[10]
        plt.figure(figsize=(13, 9))
        plt.subplot(221)
        #train[numericColumn].plot(kind='density')  # alpha=0.5 train[numericColumn].plot(kind='density')
        sns.distplot(train[numericColumn])
        #train[numericColumn].plot(kind='density')
        plt.ylabel('Density', size=10)
        plt.xlabel(numericColumn, size=10)
        plt.title("Density of " + numericColumn, size=10)

        plt.subplot(222)
        train[numericColumn].plot.hist(bins=10)  # alpha=0.5
        plt.ylabel('Count', size=10)
        plt.xlabel(numericColumn, size=10)
        plt.title("Distribution of " + numericColumn, size=10)

        plt.subplot(223)
        bp = train[numericColumn].plot.box(sym='r+', showfliers=True, return_type='dict')
        plt.setp(bp['fliers'], color='Tomato', marker='*')
        plt.ylabel('Values', size=10)
        plt.title("Spread of " + numericColumn, size=10)

        if strResponse != None:  #  strResponse = "CAT114"  # listNumericFeatures[14]
            ax = plt.subplot(224)
            if hasattr(train[strResponse], 'cat'):
                df = GetGroup_Cat_Num(train, strResponse,numericColumn, nSelectTopN = 10)
                df.plot.bar(x=strResponse, y=numericColumn, ax = ax)
                plt.ylabel(numericColumn, size=10)
                plt.xlabel(strResponse, size=10)
                plt.title("Average of "+ numericColumn + " vs " + strResponse + " (Response variable)", size=10)
            elif np.issubdtype(train[strResponse].dtype, np.number):
                train.plot.scatter(x=numericColumn, y=strResponse, ax = ax)
                plt.ylabel(strResponse, size=10)
                plt.xlabel(numericColumn, size=10)
                plt.title(numericColumn + " vs " + strResponse + " (Response variable)", size=10)
            else:
                print("Not implemented")

        plt.suptitle('Distribution of ' + numericColumn + " (Skew:" + str(round(float(train[numericColumn].skew()), 2)) + ")", size=12) # numericColumn="CONT1"
        #plt.show()
        plt.tight_layout()  # To avoid overlap of subtitles
        plt.savefig(folderImageDescriptive + "SingleNum_" + numericColumn + ".png", bbox_inches='tight')
        pdf.savefig()
        plt.close()
        plt.gcf().clear()

    pdf.close()

    return(True)  # Desc_Numeric_Single

# This provides descriptive Analytics for two numeric features
def Desc_Numeric_Double(train, listNumericFeatures1 = None, listNumericFeatures2 = None, strResponse = None, filePdf = "Desc_Numeric_Double.pdf", folderImageDescriptive = "Images/Descriptive/"):

    if listNumericFeatures1 == None:
        listNumericFeatures1 = list(train.select_dtypes(include=[np.number]).columns.values)

    if listNumericFeatures2 == None:
        listNumericFeatures2 = list(train.select_dtypes(include=[np.number]).columns.values)

    listNumericFeatures1 = list(set(listNumericFeatures1) | set(listNumericFeatures2))

    pdf = PdfPages(folderImageDescriptive + filePdf)

    for i in range(0, len(listNumericFeatures1)-1, 1):  # i = 0
        numericColumn1 = listNumericFeatures1[i]
        numericColumn2 = listNumericFeatures1[i+1]

        plt.figure(figsize=(13, 9))
        ax = plt.subplot(221)
        train.plot.scatter(x=numericColumn1, y=numericColumn2, ax = ax)
        plt.ylabel(numericColumn2, size=10)
        plt.xlabel(numericColumn1, size=10)
        plt.title(numericColumn1 + " vs " + numericColumn2, size=10)

        ax = plt.subplot(222)
        bp = train[[numericColumn1, numericColumn2]].plot.box(sym='r+', ax = ax, showfliers=True, return_type='dict')
        plt.setp(bp['fliers'], color='Tomato', marker='*')
        plt.ylabel('Values', size=10)
        plt.title("Spread of " + numericColumn1 + " and "+ numericColumn2, size=10)

        if strResponse != None:  #  strResponse = "CAT114"  # listNumericFeatures[14]
            if hasattr(train[strResponse], 'cat'):
                ax = plt.subplot(223)
                df = GetGroup_Cat_Num(train, strResponse,numericColumn1, nSelectTopN = 10)
                df.plot.bar(x=strResponse, y=numericColumn1, ax = ax)
                plt.ylabel(numericColumn1, size=10)
                plt.xlabel(strResponse, size=10)
                plt.title("Average of "+ numericColumn1 + " vs " + strResponse + " (Response variable)", size=10)

                ax = plt.subplot(224)
                df = GetGroup_Cat_Num(train, strResponse,numericColumn2, nSelectTopN = 10)
                df.plot.bar(x=strResponse, y=numericColumn2, ax = ax)
                plt.ylabel(numericColumn2, size=10)
                plt.xlabel(strResponse, size=10)
                plt.title("Average of "+ numericColumn2 + " vs " + strResponse + " (Response variable)", size=10)

            elif np.issubdtype(train[strResponse].dtype, np.number):
                ax = plt.subplot(223)
                train.plot.scatter(x=numericColumn1, y=strResponse, ax = ax)
                plt.ylabel(strResponse, size=10)
                plt.xlabel(numericColumn1, size=10)
                plt.title(numericColumn1 + " vs " + strResponse + " (Response variable)", size=10)

                ax = plt.subplot(224)
                train.plot.scatter(x=numericColumn2, y=strResponse, ax = ax)
                plt.ylabel(strResponse, size=10)
                plt.xlabel(numericColumn2, size=10)
                plt.title(numericColumn2 + " vs " + strResponse + " (Response variable)", size=10)
            else:
                print("Not implemented")

        plt.suptitle('Distribution of ' + numericColumn1 + " vs " + numericColumn2 + ". Correlation : " + str(round(float(train[[numericColumn1, numericColumn2]].corr().iloc[0,1]),2)), size=12)
        plt.tight_layout()  # To avoid overlap of subtitles
        #plt.show()
        plt.savefig(folderImageDescriptive + "DoubleNum_" + numericColumn1 + "_" + numericColumn2 + ".png", bbox_inches='tight')
        pdf.savefig()
        plt.close()
        plt.gcf().clear()
        del(numericColumn1, numericColumn2, ax)

    pdf.close()

    # Try pairplot too
    return(True)  # Desc_Numeric_Double

# This provides descriptive Analytics for single Categorical features
def Desc_Categorical_Single (train, listCategoricalFeatures = None, strResponse = None, filePdf = "Desc_Categorical_Single.pdf", folderImageDescriptive = "Images/Descriptive/"):

    if listCategoricalFeatures == None:
        listCategoricalFeatures = []
        for columnName in train.columns:
            if hasattr(train[columnName], 'cat'):
                listCategoricalFeatures.append(columnName)

    pdf = PdfPages(folderImageDescriptive + filePdf)

    for catColumn in listCategoricalFeatures:  # catColumn = listCategoricalFeatures[10]
        plt.figure(figsize=(13, 9))
        ax = plt.subplot(221)
        df = GetGroup_Cat(train, catColumn)
        df.plot.bar(x=catColumn, y="Count", ax = ax)
        plt.ylabel("Count", size=10)
        plt.xlabel(catColumn, size=10)
        plt.title("Count of "+ catColumn , size=10)

        if strResponse != None:  #  strResponse = "CAT114"  # listNumericFeatures[14]
            ax = plt.subplot(222)
            if hasattr(train[strResponse], 'cat'):
                df = pd.crosstab(train[catColumn], train[strResponse])
                df.plot(kind='bar', stacked=True, ax = ax)

            elif np.issubdtype(train[strResponse].dtype, np.number): # strResponse="CONT14"
                df = GetGroup_Cat_Num(train, catColumn,strResponse, nSelectTopN = 10)
                df.plot.bar(x=catColumn, y=strResponse, ax = ax)
                plt.ylabel(strResponse, size=10)
                plt.xlabel(catColumn, size=10)
                plt.title("Average of "+ strResponse + " (Response variable)" + " vs " + catColumn , size=10)

            else:
                print("Not implemented")

        plt.suptitle('Distribution of ' + catColumn, size=12)
        #plt.show()
        plt.tight_layout()  # To avoid overlap of subtitles
        plt.savefig(folderImageDescriptive + "SingleCat_" + catColumn + ".png", bbox_inches='tight')
        pdf.savefig()
        plt.close()
        plt.gcf().clear()
        del(ax, df)

    pdf.close()

    return(True)  # Desc_Categorical_Single

# This provides descriptive Analytics for double categorical features
def Desc_Categorical_Double (train, listCategoricalFeatures1 = None, listCategoricalFeatures2 = None, strResponse = None, filePdf = "Desc_Categorical_Double.pdf", folderImageDescriptive = "Images/Descriptive/"):

    if listCategoricalFeatures1 == None:
        listCategoricalFeatures1 = []
        for columnName in train.columns:
            if hasattr(train[columnName], 'cat'):
                listCategoricalFeatures1.append(columnName)

    if listCategoricalFeatures2 == None:
        listCategoricalFeatures2 = []
        for columnName in train.columns:
            if hasattr(train[columnName], 'cat'):
                listCategoricalFeatures2.append(columnName)

    listCategoricalFeatures1 = list(set(listCategoricalFeatures1) | set(listCategoricalFeatures2))

    pdf = PdfPages(folderImageDescriptive + filePdf)

    for i in range(0, len(listCategoricalFeatures1)-1, 1):  # i = 0
        catColumn1 = listCategoricalFeatures1[i]
        catColumn2 = listCategoricalFeatures1[i+1]

        plt.figure(figsize=(13, 9))
        ax = plt.subplot(221)
        df = pd.crosstab(train[catColumn1], train[catColumn2])
        df.plot(kind='bar', stacked=True, ax = ax)
        plt.title('Distribution of ' + catColumn1+ " vs " + catColumn2, size=12)

        if strResponse != None:
            pass

        plt.tight_layout()  # To avoid overlap of subtitles
        plt.savefig(folderImageDescriptive + "DoubleCat_" + catColumn1 + "_" + catColumn2 + ".png", bbox_inches='tight')
        pdf.savefig()
        plt.close()
        plt.gcf().clear()
        del(catColumn1, catColumn2, ax, df)

    pdf.close()

    return(True)  # Desc_Categorical_Double

# This provides descriptive Analytics for all numeric features at once
def Desc_Numeric_AllatOnce(train, listNumericFeatures = None, strResponse = None, filePdf = "Desc_Numeric_AllAtOnce.pdf", folderImageDescriptive = "Images/Descriptive/", folderOutput = "Images/Descriptive/"):

    # If incoming value is null then get all numeric from data
    if listNumericFeatures == None:
        listNumericFeatures = train.select_dtypes(include=[np.number]).columns.values

    # strip any spaces
    listNumericFeatures = list(map(str.strip, listNumericFeatures))

    # Remove response variable if present
    if strResponse != None:  # strResponse = "LOSS"
        listNumericFeatures = list(set(listNumericFeatures) - set([strResponse]))

    # Open pdf to save
    pdf = PdfPages(folderImageDescriptive + filePdf)

    # Generate paramerter for bucketting
    nTotalFeaturesCount = len(listNumericFeatures)
    nBucket = 10

    # Plot box plot
    for countBucket, startBucket in enumerate(range(0, nTotalFeaturesCount-1, nBucket)):  # startBucket = 0
        numericColumns = listNumericFeatures[startBucket:min(startBucket + nBucket, nTotalFeaturesCount)]

        plt.figure(figsize=(13, 9))
        bp = train[numericColumns].plot.box(sym='r+', showfliers=True, return_type='dict')
        plt.setp(bp['fliers'], color='red', marker='*')
        plt.ylabel('Values', size=10)
        plt.title("Conbined plot of Spread of numeric Columns")

        plt.tight_layout()  # To avoid overlap of subtitles
        plt.savefig(folderImageDescriptive + "AllAtOnceNum_" + str(countBucket+1) + ".png", bbox_inches='tight')
        pdf.savefig()
        plt.close()
        plt.gcf().clear()
        del(numericColumns, bp)
    # End of for countBucket

    # Correlation amoung numeric features
    df = train[listNumericFeatures].corr()
    print("Correlation file is saved to " + folderOutput +"AllAtOnceNum_Correlation.csv")
    df.to_csv(folderOutput + "AllAtOnceNum_Correlation.csv", index=True)

    # Draw the heatmap
    sns.heatmap(df,annot=False,  cbar=True)
    #plt.imshow(df, cmap='hot', interpolation='nearest')
    plt.title("Correlation of numeric Columns")
    plt.tight_layout()  # To avoid overlap of subtitles
    plt.savefig(folderImageDescriptive + "AllAtOnceNum__Correlation.png", bbox_inches='tight')
    pdf.savefig()
    plt.close()
    plt.gcf().clear()

    # Strong correlation
    # Set the threshold to select only highly correlated attributes
    threshold = 0.5

    #Search for the highly correlated pairs
    for i in range(0,len(df)):
        for j in range(0,len(df.columns)):
            if (abs(df.iloc[i,j]) < threshold):
                df.iloc[i,j] = 0.0

    # Draw the heatmap
    sns.heatmap(df,annot=True,  cbar=True, mask=(df==0))
    #plt.imshow(df, cmap='hot', interpolation='nearest')
    plt.title("Strong Correlation of numeric Columns")
    plt.tight_layout()  # To avoid overlap of subtitles
    plt.savefig(folderImageDescriptive + "AllAtOnceNum__Correlation_Strong.png", bbox_inches='tight')
    pdf.savefig()
    plt.close()
    plt.gcf().clear()

    #Skewness of features
    df = pd.DataFrame(train[listNumericFeatures].skew()).reset_index()
    df.columns = ["Features", "Skew"]
    df.sort_values("Skew", ascending=False, inplace=True)
    print("Skew file is saved to " + folderOutput +"AllAtOnceNum_Skew.csv")
    df.to_csv(folderOutput + "AllAtOnceNum_Skew.csv", index=False)

    pdf.close()

    return(True)  # Desc_Numeric_AllatOnce

#Dummy Encoding of categorical data, scale and center numerical data
def Encoding(train, strResponse, scale_and_center = False, fileTrain = "train_EncodedScaled.csv", fileTest = "test_EncodedScaled.csv"):
    from sklearn import preprocessing

    # get all numeric features
    listNumericFeatures = train.select_dtypes(include=[np.number]).columns.values

    # get all categorical features
    listCategoricalFeatures = []
    for columnName in train.columns:
        if hasattr(train[columnName], 'cat'):
            listCategoricalFeatures.append(columnName)

    # Remove Response variable
    if hasattr(train[strResponse], 'cat'):
        listCategoricalFeatures = list(set(listCategoricalFeatures) - set([strResponse]))
    elif np.issubdtype(train[strResponse].dtype, np.number):
        listNumericFeatures = list(set(listNumericFeatures) - set([strResponse]))

    # If scale and center (in 0-1) is true
    if (scale_and_center and len(listNumericFeatures) > 0):
        print("Scaling and Creating Dummy features")
        # get the same sclaer for both train and test
        min_max_scaler = preprocessing.MinMaxScaler()

        # do for train
        df = pd.DataFrame(min_max_scaler.fit_transform(train[listNumericFeatures]), index=train.index, columns=listNumericFeatures)

        if(len(listCategoricalFeatures) > 0):
            df = pd.concat([pd.get_dummies(train[listCategoricalFeatures]), df, train[strResponse]], axis = 1, join = "outer") # cbind
        else:
            df = pd.concat([df, train[strResponse]], axis = 1, join = "outer") # cbind

        print("Encoded train file is saved to " + fileTrain)
        df.to_csv(fileTrain, index=False)  #min_max_scaler.scale_  min_max_scaler.min_
        del(df)
    elif(len(listCategoricalFeatures) > 0):  # Make sure there are at least one categorical value
        print("Creating Dummy features only")
        # Non scaled version of above
        if len(listNumericFeatures) > 0:
            df = pd.concat([pd.get_dummies(train[listCategoricalFeatures]), train[listNumericFeatures], train[strResponse]], axis = 1, join = "outer") # cbind
        else:
            df = pd.concat([pd.get_dummies(train[listCategoricalFeatures]), train[strResponse]], axis = 1, join = "outer") # cbind

        print("Encoded train file is saved to " + fileTrain)
        df.to_csv(fileTrain, index=False)  #   min_max_scaler.scale_        min_max_scaler.min_
        del(df)

    return(True)
    # Encoding end

#Dummy Encoding of categorical data, scale and center numerical data
def ScaleAndCenter_NumericOnly(train, strResponse = None):
    from sklearn import preprocessing

    # get all numeric features
    listNumericFeatures = train.select_dtypes(include=[np.number]).columns.values

    # Remove Response variable
    if strResponse != None:
        if np.issubdtype(train[strResponse].dtype, np.number):
            listNumericFeatures = list(set(listNumericFeatures) - set([strResponse]))

    listRemainingFeatures = list(set(train.columns) - set(listNumericFeatures))

    # If scale and center (in 0-1) is true
    if (len(listNumericFeatures) > 0):
        print("Scaling features")
        # get the same sclaer for both train and test
        min_max_scaler = preprocessing.MinMaxScaler()

        # do for train
        df = pd.DataFrame(min_max_scaler.fit_transform(train[listNumericFeatures]), index=train.index, columns=listNumericFeatures)

        if(len(listRemainingFeatures) > 0):
            df = pd.concat([df, train[listRemainingFeatures]], axis = 1, join = "outer") # cbind
    del(listNumericFeatures, listRemainingFeatures)

    return(df)
    # ScaleAndCenter_NumericOnly end

def reduce_concat(x, sep=""):
    import functools
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

# Save result
def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)

#Description: Get PCA transformed data.
# Input: provide non scaled data and count of PCA component
def GetPCA(train, n_components=None, bScale = False, fileImageToSave = "pca.png"):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Scale the incoming data
    if bScale:
        train = ScaleAndCenter_NumericOnly(train)

    # create instance of PCA object
    pca = PCA(n_components=n_components)
    # Fit the model with X and apply the dimensionality reduction on X.
    train = pca.fit_transform(train)

    #Cumulative Variance explains
    cumVarExplained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    n_components = len(cumVarExplained)

    #Plot the cumulative explained variance as a function of the number of components
    plt.subplots(figsize=(13, 9))
    plt.plot(range(1,n_components+1,1), cumVarExplained, 'bo-')  #
    plt.ylabel('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(1, n_components+1, 1))
    plt.title("PCA: Number of Features vs Variance (%)")
    #plt.ylim([0.0, 1.1])
    #plt.yticks(np.arange(0.0, 1.1, 0.10))
    #plt.show()
    plt.tight_layout()  # To avoid overlap of subtitles
    print("PCA: Number of Features vs Variance (%) is saved to " + fileImageToSave)
    plt.savefig(fileImageToSave, bbox_inches='tight')
    plt.close()
    plt.gcf().clear()

    print("PCA: Number of Features vs Variance (%) CSV is saved to pca.csv")
    df = pd.DataFrame({"CumVarExplained" : cumVarExplained, "Components": range(1,n_components+1,1)})
    df.to_csv('pca.csv', index=False)
    del(df)
    # Prepapre for return
    train = pd.DataFrame(train, columns= paste(["PC"] * n_components, np.arange(1, n_components+1, 1), sep=''))  #   # ('string\n' * 4)[:-1]

    return(train.iloc[:, 0:n_components])
    # end of GetPCA

#Description: It founds out outlier by at least two algorithum. Train data need be passsed without response variable
def GetOutliersIndex_by_OneClassSVM_RobustCovariance_IsolationForest(train, nTopOutliers = 5):
    # Libraries
    from scipy import stats
    import matplotlib.font_manager
    from sklearn import svm
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest

    # Get fraction of outliers
    outliers_fraction = min(nTopOutliers/train.shape[0], 0.5)

    # define outlier detection tools to be compared
    classifiers = {
    "OneClassSVM": svm.OneClassSVM(nu=0.25, kernel="rbf", gamma=max(0.1, 1/train.shape[1])),
    "RobustCovariance": EllipticEnvelope(contamination=outliers_fraction),
    "IsolationForest": IsolationForest(max_samples=train.shape[0], contamination=outliers_fraction)}

    # Fit the model with varying cluster separation
    df = pd.DataFrame()
    for i, (clf_name, clf) in enumerate(classifiers.items()):  # classifiers.items()[0]
        clf.fit(train)  # fit the data and tag outliers
        scores_pred = clf.decision_function(train) # Average anomaly score of X of the base classifiers.
        df = pd.concat([df, pd.DataFrame(scores_pred.astype(np.number))], axis = 1, join = "outer") # cbind

    # set the column name
    df.columns = classifiers.keys()
    #df.loc[:,"OneClassSVM"] = df.loc[:,"OneClassSVM"].astype(np.number)

    # To be safe. convert the column to numeric
    #df[df.columns] = df[df.columns].apply(pd.to_numeric)
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    # Get index as one column so that it will help in merge
    df = df.reset_index()

    # For each algorithum get the top rank of outliers
    for column in classifiers.keys():  # column = "IsolationForest"
        # First sort the score in asending order. Less score is highest outlier
        df.sort_values(column, ascending=True, inplace=True)
        # Get originla index and outlier index mapping
        recode = {}
        for rnk in range(0,df.shape[0],1):  # rnk = 0
            recode[int(df.iloc[rnk]["index"])] = rnk+1
        # Create new column with sorted index.
        df[str(column + "_rank")] = df["index"].map(recode)

    # Get back the score data in original order
    df.sort_values("index", ascending=True, inplace=True)
    # Now drop index column
    df = df.drop("index", axis  = 1)

    # Now keep new sorted columns only
    # rank_cols = [col for col in df.columns if '_rank' in col]
    df = df.filter(regex='_rank')

    # Calculate outlier score. Logic is as follows - 1. Take sum of all columns of each row. 2. Substarct max value so that only minimum is taken
    # 3. Now take average of score as final score
    fScoreMean = (df.sum(axis=1, skipna=True, numeric_only=True) - df.max(axis=1, skipna=True, numeric_only=True))/(df.shape[1]-1)

    # Make data frame for easy redability
    df = pd.DataFrame({"ScoreMean" : fScoreMean})
    #df.columns = ["ScoreMean"]
    # Sort and keep smallest on top
    df.sort_values("ScoreMean", ascending=True, inplace=True)

    #Take top nTopOutliers as outlier
    listOutliers = df.index[0:nTopOutliers]

    return(listOutliers)
    # end of GetOutliersIndex_by_OneClassSVM_RobustCovariance_IsolationForest

#Description: Now plot the outlier againt whole data set
def PlotOutlierByTakingTwoPCA(train, listOutliers, bScale = False, fileImageToSave = None, strNonOutlierColor = "lightblue", strOutlierColor = "red"):
    import matplotlib.pyplot as plt

    # First take PCA with 2 component only
    train = GetPCA(train, 2, bScale)
    # Color code the normal to black and outlier to red
    train["Color"] = strNonOutlierColor
    train.loc[listOutliers, "Color"] = strOutlierColor
    #colors = np.where(df.col3 > 300, 'r', 'k')
    train[train.Color == "lightblue"]

    # Draw the graph and save
    plt.subplots(figsize=(13, 9))
    #train.plot.scatter(x="PC1", y="PC2", color = train.Color)
    plt.scatter(x=train[train.Color == strNonOutlierColor].PC1, y=train[train.Color == strNonOutlierColor].PC2, c= train[train.Color == strNonOutlierColor].Color)
    plt.scatter(x=train[train.Color == strOutlierColor].PC1, y=train[train.Color == strOutlierColor].PC2, c= train[train.Color == strOutlierColor].Color)
    plt.ylabel("PC2", size=10)
    plt.xlabel("PC1", size=10)
    plt.title("PCA transformed graph", size=10)
    plt.tight_layout()  # To avoid overlap of subtitles

    if fileImageToSave != None:
        print("Ouliers: PCA transformed graph is saved to " + fileImageToSave)
        plt.savefig(fileImageToSave, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    plt.gcf().clear()

    return
    # End of PlotOutlierByTakingTwoPCA

# Description: Plot average distance from observations from the cluster centroid to use the Elbow Method to identify number of clusters to choose
def GetElbowImageForKClusters(train, bScale = False, fileElbowImageToSave = None, rangeKClusters = range(1,10)):
    # Library
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans

    # If scalling and centering required
    if(bScale):
        train = ScaleAndCenter_NumericOnly(train)

    # Calculate distances bewteen clusters
    lsitMeandist=[]
    for k in rangeKClusters:
        model=KMeans(n_clusters=k)
        model.fit(train)
        clusassign=model.predict(train)
        lsitMeandist.append(sum(np.min(cdist(train, model.cluster_centers_, 'euclidean'), axis=1))/ train.shape[0])

    # Plot the K in Elbow image
    plt.plot(rangeKClusters, lsitMeandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')

    plt.tight_layout()  # To avoid overlap of subtitles

    if fileElbowImageToSave != None:
        print("K with the Elbow graph is saved to " + fileElbowImageToSave)
        plt.savefig(fileElbowImageToSave, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
    plt.gcf().clear()

    return
    # end of GetElbowImageForKClusters

#Put Cluster's attribute with content in specified file
#count_cluster = 3; col_cluster = "CLUSTERID"; FeaturesForCluster = train.columns.values; HeaderPrefix = "Cluster";TopNFactorCount = 5; listCoreExcluded = FeaturesForCluster
def getClusterContentWithMaxPercentageAttr(train, count_cluster, col_cluster, FeaturesForCluster, HeaderPrefix = "Cluster",TopNFactorCount = 1, listCoreExcluded = None):
    from scipy.interpolate import interp1d

    # Create Data table with structure - Cluster1..n
    col_names = paste([HeaderPrefix] * count_cluster, np.arange(1, count_cluster+1, 1), sep='')
    dt_cluster = pd.DataFrame(columns= col_names)

    # Store size of cluster
    dtClusterSize = GetGroup_Cat(train, col_cluster)
    dtClusterSize[col_cluster] = dtClusterSize[col_cluster].astype(np.int)
    dtClusterSize.drop(['CountPercentage'], axis=1, inplace=True)
    dtClusterSize.sort_values([col_cluster], inplace=True)

    FeaturesForCluster_Trimmed = FeaturesForCluster = list(set(FeaturesForCluster) - set([col_cluster]))

    # Iterate for each factor and get the frequencies
    for col_cluster_feature in FeaturesForCluster: # col_cluster_feature = "SPECIES"
      if train[col_cluster_feature].dtype.name == 'category' or  train[col_cluster_feature].dtype.name == 'bool': # & does row-wise and for the two series.
          # Get table for factor feature w.r.to cluster
          fac_data = []
          tab_attr_cluster = get_group_cat_many(train, [col_cluster_feature, col_cluster], bCountPercentage = False)
          for cluster_id in np.arange(1, count_cluster+1, 1): # cluster_id = 1
              df = tab_attr_cluster[tab_attr_cluster[col_cluster] == cluster_id]
              df = df[0:min(df.shape[0],TopNFactorCount)]
              fac_data.append(','.join(paste(df[col_cluster_feature], sep="")))
              del(df)
          df = pd.DataFrame(fac_data).transpose(); df.columns = col_names
          dt_cluster = pd.concat([dt_cluster, df], axis = 0) # rbind
          del(df, tab_attr_cluster, fac_data)
      elif np.issubdtype(train[col_cluster_feature].dtype, np.number) or np.issubdtype(train[col_cluster_feature].dtype, np.int): # col_cluster_feature = "SEPAL.LENGTH"
        # Create table of col cluster and mean of feature colume
        tab_attr_cluster = GetGroup_Cat_Num(train, col_cluster,col_cluster_feature)
        tab_attr_cluster.sort_values(col_cluster, inplace = True)

        # To interpolate the quantile, code is taken from https://stackoverflow.com/questions/26489134/whats-the-inverse-of-the-quantile-function-on-a-pandas-series
        # set up a sample dataframe
        df = pd.DataFrame(train[col_cluster_feature])

        # sort it by the desired series and caculate the percentile
        df = df.sort_values(col_cluster_feature).reset_index()
        df['Q'] = df.index / float(len(df) - 1); df['Q'] = round(df['Q'], 2)

        # setup the interpolator using the value as the index
        interp = interp1d(df[col_cluster_feature], df['Q'])
        # Test
        # df[col_cluster_feature].quantile(0.5) # 5.8
        # interp(5.8) # 4899328859060403
        tab_attr_cluster['Q'] = interp(tab_attr_cluster[col_cluster_feature])
        tab_attr_cluster['Q'] = tab_attr_cluster['Q'].round(2)
        #tab_attr_cluster['MEAN_Q'] = tab_attr_cluster[[col_cluster_feature, 'Q']].apply(lambda x: ''.join(str(x)), axis=1)
        tab_attr_cluster['MEAN_Q'] = tab_attr_cluster[col_cluster_feature].astype(str) + "," + tab_attr_cluster['Q'].astype(str)

        df = pd.DataFrame(tab_attr_cluster['MEAN_Q']).transpose(); df.columns = col_names
        dt_cluster = pd.concat([dt_cluster, df], axis = 0)

        del(df, interp, tab_attr_cluster)
      else:
        FeaturesForCluster_Trimmed <- list(set(FeaturesForCluster_Trimmed) - set([col_cluster_feature]))
        print("Not implemented for ", col_cluster_feature, train[col_cluster_feature].dtype)

    # Clear the index before concate
    dt_cluster.reset_index(drop = True, inplace = True)

    # Add size of each clsuter
    df = pd.DataFrame(dtClusterSize['Count']).transpose(); df.columns = col_names
    dt_cluster = pd.concat([dt_cluster, df], axis = 0)
    dt_cluster.reset_index(drop = True, inplace = True)

    # Add feature columns
    df = pd.concat([pd.DataFrame({'Features' : FeaturesForCluster_Trimmed}), pd.DataFrame({'Features' : ['SIZE']})], axis = 0)
    df.reset_index(drop = True, inplace = True)
    dt_cluster = pd.concat([df, dt_cluster], axis = 1)

# Add Core column to identify the feature that were part of cluster formation
    if listCoreExcluded is not None: # listCoreExcluded = None
        df = pd.DataFrame({'Core' : [feature in listCoreExcluded for feature in dt_cluster.Features]})
        dt_cluster = pd.concat([df, dt_cluster], axis = 1)
        #dt_cluster.loc[dt_cluster['Features'] == 'SIZE','Core'] = df[df['Core'] == True].shape[0]
        cols = dt_cluster.columns.tolist()
        temp = cols[0]; cols[0] = cols[1]; cols[1] = temp
        dt_cluster = dt_cluster[cols]
        del(df, cols, temp)

    del(col_names, dtClusterSize, FeaturesForCluster_Trimmed)

    return(dt_cluster)
   # end of getClusterContentWithMaxPercentageAttr
#df = getClusterContentWithMaxPercentageAttr(train, count_cluster = 3, col_cluster = 'CLUSTERID', FeaturesForCluster = train.columns.values, HeaderPrefix = "Cluster",TopNFactorCount = 2, listCoreExcluded = train.columns.values)

#https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn
def mean_absolute_percentage_error(y_true, y_pred):

    error = 100.0
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # The follwoing may divide by 0 and hence through exception. Hence by default MAE will be returned
        error = np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    except:
        g_logger.error("Unexpected error and hence returning MAE:", sys.exc_info()[0])
        error = np.round(mean_absolute_error(y_true, y_pred), 2)

    return error
# mean_absolute_percentage_error

# actual = ar_data_y;seriesResuduals= actual - pred
def ResidualPlot(actual, pred, fileImageToSave = "Residual.png"):
    import statsmodels.api as sm

    actual = np.ravel(np.array(actual)); pred = np.ravel(np.array(pred))

    if type(actual) != np.ndarray or type(pred) != np.ndarray:
        print("Type of actual or  pred is not 'np.ndarray'. Not plotting")
        return

    # Basic error to put on super title
    mae = np.round(mean_absolute_error(actual, pred), 2)
    rmse = np.round(np.sqrt(mean_squared_error(actual, pred)), 2)
    mape = mean_absolute_percentage_error(actual, pred)
    sup_title = 'Residual  anlaysis-> mae:' + str(mae) + ', rmse:' + str(rmse) + ', mape:' + str(mape)

    # REsidual as few places required
    seriesResuduals= actual - pred

    # DL is sending in ndarray and hence taking first column only
    if type(seriesResuduals) == np.ndarray:
        seriesResuduals = np.reshape(seriesResuduals, (seriesResuduals.shape[0], 1))
        seriesResuduals = seriesResuduals[:,0]

    # DW test
    from statsmodels.stats.stattools import durbin_watson
    msg = "This statistic will always be between 0 and 4. 2 - indicating no serial correlation. The closer to 0 the statistic, the more evidence for positive serial correlation. The closer to 4, the more evidence for negative serial correlation."
    stat_durbin_watson = round(durbin_watson(seriesResuduals),2)
    print("Durbin_Watson:" + str(stat_durbin_watson) + "->" + msg)

    fig = plt.figure(figsize=(13, 9))

    ax = plt.subplot(221)
    sm.qqplot(seriesResuduals, ax =  ax, line='45')
    plt.title("Residuals: Expected Line plot")

    ax = plt.subplot(222)
    #sm.qqplot(seriesResuduals, ax =  ax, line='r')
    #plt.title("Residuals: Line plot within Residuals")
    sns.distplot(seriesResuduals)
    plt.title("Residuals: Normality view")

    # scatter plot of residuals
    ax = plt.subplot(223)
    plt.scatter(x=range(1, len(seriesResuduals)+1), y=seriesResuduals) # , ax = ax
    #plt.xticks(np.arange(1, len(seriesResuduals)+1, 1))
    #plt.xticks(range(1, len(seriesResuduals)+1))
    plt.axhline(y=0, color='r')
    plt.ylabel('Standardized Residual')
    plt.xlabel('Observation Number')
    plt.title("Residuals: Scatter plot")
    #plt.show()

    # scatter plot of actual and pred
    ax = plt.subplot(224)
    plt.scatter(range(len(actual)), actual,  color='red', marker = '*', s = 30,label='Original')
    plt.scatter(range(len(pred)), pred,  color='green', marker = '*', s = 30,label='Predicted')
    plt.legend(loc='best')
    plt.title(''.join(["Scatter plot DW(", str(stat_durbin_watson),"): actual (red) and pred (green)"]))

    plt.suptitle(sup_title, size=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  #fig , rect=[0, 0.03, 1, 0.95] To avoid overlap of subtitles
    print("Residuals: Images is saved to " + fileImageToSave)
    plt.savefig(fileImageToSave, bbox_inches='tight')
    plt.close()
    plt.gcf().clear()

    return mae, rmse, mape
    # end of ResidualPlot
