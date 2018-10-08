#cox model trained in NLST and test in TCGA
library(survival)
library(glmnet)
setwd("//198.215.54.48/swan15/achen/survival_analysis")
filename <- 'region_properties_slides_all_Shidan_v2.csv'

##### NLST: training model #####
# Please refer to 5_univariateAnalysisSlides_v3.R
# # Read in data
# dat <- read.csv(paste("../1_NLST_heatmap/", filename, sep = ""))
# 
# # Univariate CoxPH regression analysis of covariates
# unicox_df <- data.frame(var=NA, HR=NA, se=NA, z=NA, pv=NA) # variable, hazards ratio, standard error, z, p-value
# total_features = dim(dat)[2]
# colnames = colnames(dat)[10:total_features] # remove slide ID, tissue ID, time, status, clincical variables
# colnames <- colnames[-(60:62)]
# 
# # Combine same patient
# dat_combine <- dat[which(dat$tissue_id == 0), ]
# pID <- unique(dat$slide_id)
# for (i in 1:length(pID)){
#   thisPatient <- which(dat$slide_id == pID[i])
#   if (length(thisPatient) == 1){
#     next
#   }else{
#     dat_combine[which(dat_combine$slide_id == pID[i]), colnames] <- colMeans(dat[thisPatient, colnames], na.rm = T)
#   }
# }
# 
# # Set group = tumor_percent < 0.7, and only keep the first tissue patch
# #group <- which(dat_combine$tumor_percent <= 0.8)
# group <- 1:dim(dat_combine)[1]
# 
# # Univariate cox model
# for(i in 1:length(colnames)){
#   var = colnames[i]
#   cox = coxph(Surv(time, status==1) ~ dat_combine[group, var], data = dat_combine[group, ])
#   unicox_df[i,'var'] = var
#   unicox_df[i,2:5] = summary(cox)$coef[c(2:5)]
# }
# unicox_df

# select significant vars.
features <- unicox_df$var
#0.0007 = 0.05/72 performs well
#sig_features <- features[which(unicox_df$pv <= 0.001)]
#sig_features <- sig_features[12:24]
sig_features <- features[which(unicox_df$pv <= 0.05)]
sig_features <- sig_features[c(20:28, 30:35)]
#sig_features <- sig_features[c(25:32, 34:39)]
#sig_features <- sig_features[15:28]
#sig_features <- sig_features[c(1:2, 15:30)] #0.001
#sig_features <- sig_features[c(1:2, 25:43)] #0.05
#sig_features <- sig_features[c(1:2, 14:27)] #0.0007
surv <- Surv(dat_combine[, "time"], dat_combine[, "status"])
model <- coxph(surv ~ ., data = dat_combine[, sig_features])

## save risk score and clinical data to csv
to.save <- data.frame(matrix(NA, nrow = 150, ncol = 8))
colnames(to.save) <-  c("pID", "age", "gender", "tobacco", "stage", "grade", "time", "event")
to.save$pID <- NLST_pID_unique
to.save$age <- dat_combine$age
to.save$gender <- dat_combine$gender
to.save$tobacco <- dat_combine$tobacco
to.save$stage <- dat_combine$stage
to.save$grade <- dat_combine$grade
to.save$time <- surv[, 1]
to.save$event <- surv[, 2]
# write.csv(to.save, file = "clinical_vars_NLST.csv", row.names = F)

#********************
#*****  glmnet  *****
#********************
group <- 1:150
cox.temp <- coxph(surv ~ ., data = dat_combine[group, sig_features])
to.remove <- cox.temp$na.action
if (length(to.remove) > 0){
  model2 <- glmnet(x = as.matrix(dat_combine[group[-to.remove], sig_features]), y = as.matrix(surv[-to.remove, ]),
                   family = "cox", alpha = 0.5, standardize=T)
  cv.glmnet(x = as.matrix(dat_combine[group[-to.remove], sig_features]), y = as.matrix(surv[-to.remove, ]),
            family = "cox", alpha = 0.5, standardize=T)$lambda.min
}else{
  model2 <- glmnet(x = as.matrix(dat_combine[group, sig_features]), y = as.matrix(surv),
                   family = "cox", alpha = 0.5, standardize=T)
  cv.glmnet(x = as.matrix(dat_combine[group, sig_features]), y = as.matrix(surv),
            family = "cox", alpha = 0.5, standardize=T)$lambda.min
}
coef(model2, s = c(0.01, 0.014, 0.02, 0.022, 0.03, 0.04))
# Save model
save(model2, sig_features, file = "glmnet_model_v3.RData")

##### TCGA: testing model #####
# Read in data
dat_TCGA <- read.csv(paste("../2_TCGA_heatmap/", filename, sep = ""), stringsAsFactors = F)

# Combine same patient
dat_combine_TCGA <- data.frame(matrix(data = NA, ncol = dim(dat_TCGA)[2]))
colnames(dat_combine_TCGA) <- colnames(dat_TCGA)
patient_id_TCGA <- sapply(dat_TCGA$slide_id, substr, 1, 12, simplify = T)
pID_TCGA <- unique(patient_id_TCGA) #628 slides, 390 patients
for (i in 1:length(pID_TCGA)){
  thisPatient <- which(patient_id_TCGA == pID_TCGA[i])
  dat_combine_TCGA[i, colnames] <- colMeans(dat_TCGA[thisPatient, colnames], na.rm = T)
  dat_combine_TCGA[i, 1:8] <- dat_TCGA[thisPatient[1], 1:8]
}

table(dat_combine_TCGA$stage)
dat_combine_TCGA$stage[which(dat_combine_TCGA$stage %in% c("Stage IA", "Stage IB"))] <- "Stage I"
dat_combine_TCGA$stage[which(dat_combine_TCGA$stage %in% c("Stage IIA", "Stage IIB"))] <- "Stage II"
dat_combine_TCGA$stage[which(dat_combine_TCGA$stage %in% c("Stage IIIA", "Stage IIIB"))] <- "Stage III"
dat_combine_TCGA <- dat_combine_TCGA[which(dat_combine_TCGA$stage %in% c("Stage I", "Stage II", "Stage III", "Stage IV")), ]

surv_TCGA <- Surv(dat_combine_TCGA$time, dat_combine_TCGA$status)

#### Check p/a^2
paRatio <- dat_combine_TCGA$total_perimeter_2^2/dat_combine_TCGA$total_filled_area_2
dat_combine_TCGA$total_pa_ratio_2 <- paRatio
paRatio <- dat_combine_TCGA$main_perimeter_2^2/dat_combine_TCGA$main_filled_area_2
dat_combine_TCGA$main_pa_ratio_2 <- paRatio
coxph(surv_TCGA ~ paRatio, data = dat_combine_TCGA)

# Set group = tumor_percent < 0.7, and only keep the first tissue patch
#group_TCGA <- which(dat_combine_TCGA$tumor_percent <= 0.5)
group_TCGA <- 1:dim(dat_combine_TCGA)[1]

# test model
#### Model 1: coxph ####
risk <- predict(model, dat_combine_TCGA)
summary(coxph(surv_TCGA ~ risk))
risk_high <- risk >= median(risk, na.rm = T)
sf <- survfit(surv_TCGA ~ risk_high)
plot(sf, col = c("green", "red"), mark = 3)
summary(coxph(surv_TCGA ~ risk_high))


#### Model 2: coxph with elastic net ####
risk <- predict(model2, newx = as.matrix(dat_combine_TCGA[, sig_features]), s = 0.014)
summary(coxph(surv_TCGA ~ risk))

#### mclust ####
library(mclust)
hist(risk, breaks = 50)
mc <- Mclust(risk, G=2, model="E")
risk_high <- mc$classification == 2
table(risk_high)

dens <- densityMclust(risk)
br <- seq(min(risk), max(risk), length = 30)
plot(dens, what = "density", data = risk, breaks = br)
dens$BIC
summary(dens, parameters = T)
#### end of mclust

#### plot for mclust ####
#BIC
BIC <- mclustBIC(risk)
#pdf("../writing/mCluster_BIC.pdf", 5, 5)
plot(BIC)
#dev.off()

source("//198.215.54.48/swan15/R/myfunctions/mclust.hist.plot.R")
#pdf("../writing/mCluster_hist.pdf", 5, 5)
mc2 <- mclust.hist.plot(risk, xlab = "Risk Score", model = "V", G = 3, breaks = 40, ylim = c(0, 0.8))
legend(1.0, 0.8, legend = c("Low Risk", "high Risk"), col = mc2[[2]], pch = 15, bty = "n")
risk_high <- mc2[[1]]$classification == 2
table(risk_high)
#dev.off()
#### end of plot for mclust 

#### figures ####
risk_high <- risk > median(risk, na.rm = T)
table(risk_high)

surv_TCGA <- Surv(dat_combine_TCGA$time, dat_combine_TCGA$status)
sf <- survfit(surv_TCGA ~ risk_high)
#source("//198.215.54.48/swan15/R/myfunctions/km.num.plot.R")
#km.num.plot(sf, label.range = c(0, 6000), label.num = 7, mycolor = c("green", "red"), 
#            mylegend = c("Low Risk", "High Risk"), legend.x = 3000, legend.y = 0.9, mark = 3)
#pdf("../writing/TCGA_model_test.pdf", 5, 5)
plot(sf, col = c("green", "red"), mark = 3, xlab = "Time (Days)", ylab = "Survival Probability")
legend(3000, 0.9, legend = c("Low Risk", "High Risk"), col = c("green", "red"), lty = 1, bty = "n")
#dev.off()
summary(coxph(surv_TCGA ~ risk))
summary(coxph(surv_TCGA ~ risk_high))
survdiff(surv_TCGA ~ risk_high)
survdiff(surv_TCGA ~ risk_high, rho = 2)

## without transfer (Stage IV); performance didn't improve
sf <- survfit(surv_TCGA[which(dat_combine_TCGA$stage != "Stage IV"), ] ~ risk_high[which(dat_combine_TCGA$stage != "Stage IV")])
plot(sf, col = c("green", "red"), mark = 3, xlab = "Time (Days)", ylab = "Survival Probability")
legend(3000, 0.9, legend = c("Low Risk", "High Risk"), col = c("green", "red"), lty = 1, bty = "n")
survdiff(surv_TCGA[which(dat_combine_TCGA$stage != "Stage IV"), ] ~ risk_high[which(dat_combine_TCGA$stage != "Stage IV")])

## multivariate analysis
summary(coxph(surv_TCGA ~ risk_high + dat_combine_TCGA$age + dat_combine_TCGA$gender + dat_combine_TCGA$tobacco + dat_combine_TCGA$stage))
summary(coxph(surv_TCGA ~ risk + dat_combine_TCGA$age + dat_combine_TCGA$gender + dat_combine_TCGA$tobacco + dat_combine_TCGA$stage))

## save risk score and clinical data to csv
to.save <- data.frame(matrix(NA, nrow = 389, ncol = 8))
colnames(to.save) <-  c("pID", "riskScore", "age", "gender", "tobacco", "stage", "time", "event")
to.save$riskScore <- risk
to.save$pID <- sapply(dat_combine_TCGA$slide_id, substr, 1, 12, simplify = T)
to.save$age <- dat_combine_TCGA$age
to.save$gender <- dat_combine_TCGA$gender
to.save$tobacco <- dat_combine_TCGA$tobacco
to.save$stage <- dat_combine_TCGA$stage
to.save$time <- surv_TCGA[, 1]
to.save$event <- surv_TCGA[, 2]
#write.csv(to.save, file = "risk_score_clinical_vars_TCGA_v2.csv", row.names = F)
