import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import re
import os
from tqdm import tqdm
from tqdm.std import trange
import warnings

def binomial_glm(df, out="freq", N="n_total_mentions"):
    df.columns = [re.sub("\W+", "_", c.strip()) for c in df.columns]
    df = df[(df[N] > 0) & (df["n_titles"] >= 30)]
    var = df.iloc[:,:-8].var()
    non_singular_columns = var.index[var > 0].tolist()
    if df["kind"].unique().size > 0:
        non_singular_columns.append("kind")
    df = df[non_singular_columns + [N, out]]
    
    if df.shape[0] >= 5 and df.shape[1] > 2:

        genres = [c for c in df.columns if c.startswith("Genre")]
        countries = [c for c in df.columns if c.startswith("Country")]
        rhs = []
        if "year" in df.columns:
            rhs.append("year")
        if "kind" in df.columns:
            rhs.append("kind")
        for genre in genres:
            rhs.append("C({})".format(genre))
        for country in countries:
            rhs.append("C({})".format(country))
        rhs = " + ".join(rhs)
        formula = "{} ~ {}".format(out, rhs)
        model = smf.glm(formula, df, family = sm.families.Binomial(), var_weights = df[N])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit()
        
        coefficients = pd.concat([result.params, result.pvalues], axis = 1)
        coefficients.columns = ["coefficient", "pvalue"]
        coefficients.index.name = "param"
        coefficients = coefficients[coefficients["pvalue"] < 0.05]
        return coefficients, result

def binomial_analysis(professions_file, profession_media_dir, soc_media_dir):
    professions_df = pd.read_csv(professions_file, index_col=None)
    professions = professions_df.profession_merge.unique()[:500]

    for profession in tqdm(professions, desc="profession"):
        profession_media_df = pd.read_csv(os.path.join(profession_media_dir, profession + ".csv"), index_col=None)
        profession_media_df["n_sentiment_mentions"] = profession_media_df["n_pos_mentions"] + profession_media_df["n_neg_mentions"]
        
        try:
            freq_result = binomial_glm(profession_media_df, out="freq", N="n_total_mentions")
            if freq_result is not None:
                freq_coefficients, freq_glm = freq_result
                freq_coefficients.to_csv(os.path.join(profession_media_dir, profession + ".freq.coeff.csv"), index=True)
                freq_glm.save(os.path.join(profession_media_dir, profession + ".freq.glm"))
        except Exception:
            pass

        try:
            sentiment_result = binomial_glm(profession_media_df, out="sentiment", N="n_sentiment_mentions")
            if sentiment_result is not None:
                sentiment_coefficients, sentiment_glm = sentiment_result
                sentiment_coefficients.to_csv(os.path.join(profession_media_dir, profession + ".sentiment.coeff.csv"), index=True)
                sentiment_glm.save(os.path.join(profession_media_dir, profession + ".sentiment.glm"))
        except Exception:
            pass

    for i in trange(23, desc="soc"):
        soc_code = 11 + 2 * i
        for n in range(1, 6):
            pth = os.path.join(soc_media_dir, "{}.{}.csv".format(soc_code, n))
            if os.path.exists(pth):
                soc_media_df = pd.read_csv(pth, index_col=None)
                soc_media_df["n_sentiment_mentions"] = soc_media_df["n_pos_mentions"] + soc_media_df["n_neg_mentions"]
        
                try:
                    freq_result = binomial_glm(soc_media_df, out="freq", N="n_total_mentions")
                    if freq_result is not None:
                        freq_coefficients, freq_glm = freq_result
                        freq_coefficients.to_csv(os.path.join(soc_media_dir, "{}.{}.freq.coeff.csv".format(soc_code, n)), index=True)
                        freq_glm.save(os.path.join(soc_media_dir, "{}.{}.freq.glm".format(soc_code, n)))
                except Exception:
                    pass

                try:
                    sentiment_result = binomial_glm(soc_media_df, out="sentiment", N="n_sentiment_mentions")
                    if sentiment_result is not None:
                        sentiment_coefficients, sentiment_glm = sentiment_result
                        sentiment_coefficients.to_csv(os.path.join(soc_media_dir, "{}.{}.sentiment.coeff.csv".format(soc_code, n)), index=True)
                        sentiment_glm.save(os.path.join(soc_media_dir, "{}.{}.sentiment.glm".format(soc_code, n)))
                except Exception:
                    pass

if __name__ == "__main__":
    professions_file = "/proj/sbaruah/subtitle/profession/csl/data/mentions/professions.word_filtered.sense_filtered.merged.csv"
    profession_media_dir = "/proj/sbaruah/subtitle/profession/csl/data/analysis_data/media_data/profession"
    soc_media_dir = "/proj/sbaruah/subtitle/profession/csl/data/analysis_data/media_data/soc"
    binomial_analysis(professions_file, profession_media_dir, soc_media_dir)