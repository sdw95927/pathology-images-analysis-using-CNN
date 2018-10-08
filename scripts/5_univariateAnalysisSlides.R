library(survival)
setwd("//198.215.54.48/swan15/achen/survival_analysis")
filename <- 'region_properties_slides_all.csv'

##### NLST #####
# Read in data
dat <- read.csv(paste("../1_NLST_heatmap/", filename, sep = ""))

# Univariate CoxPH regression analysis of covariates
unicox_df = data.frame(var=NA, HR=NA, se=NA, z=NA, pv=NA) # variable, hazards ratio, standard error, z, p-value
total_features = dim(dat)[2]
colnames = colnames(dat)[10:total_features] # remove slide ID, tissue ID, time, status, clincical variables

# Check distrbution
pdf("NLST_variable_distribution.pdf", 5, 5)
for(i in 1:length(colnames)){
  var = colnames[i]
  hist(dat[, var], main = var, breaks = 30)
}
dev.off()

# Combine same patient
dat_combine <- dat[which(dat$tissue_id == 0), ]
pID <- unique(dat$slide_id)
for (i in 1:length(pID)){
  thisPatient <- which(dat$slide_id == pID[i])
  if (length(thisPatient) == 1){
    next
  }else{
    dat_combine[which(dat_combine$slide_id == pID[i]), colnames] <- colMeans(dat[thisPatient, colnames], na.rm = T)
  }
}

# Set group = tumor_percent < 0.7, and only keep the first tissue patch
group <- which(dat_combine$tumor_percent <= 0.8)
group <- 1:dim(dat_combine)[1]

# Univariate cox model
for(i in 1:length(colnames)){
  var = colnames[i]
  cox = coxph(Surv(time, status==1) ~ dat_combine[group, var], data = dat_combine[group, ])
  unicox_df[i,'var'] = var
  unicox_df[i,2:5] = summary(cox)$coef[c(2:5)]
}
unicox_df

# Sort by p-value and identify features with p_value less than 0.05
#unicox_df = unicox_df[with(unicox_df, order(pv)),]
features = unicox_df$var
sig_features = features[with(unicox_df, pv<0.05)] 
hist(unicox_df$pv[which(!is.nan(unicox_df$pv))], breaks=seq(0,1,.05), labels=T) 

# Write to Excel file
write.csv(unicox_df, file = "univariate_patient_NLST.csv", row.names = F)


##### TCGA #####
# Read in data
dat <- read.csv(paste("../2_TCGA_heatmap/", filename, sep = ""), stringsAsFactors = F)

# Univariate CoxPH regression analysis of covariates
unicox_df <- data.frame(var=NA, HR=NA, se=NA, z=NA, pv=NA) # variable, hazards ratio, standard error, z, p-value
total_features <- dim(dat)[2]
colnames <- colnames(dat)[9:total_features] # remove slide ID, tissue ID, time, status, clincical variables

# Check distrbution
pdf("TCGA_variable_distribution.pdf", 5, 5)
for(i in 1:length(colnames)){
  var <- colnames[i]
  hist(dat[, var], main = var, breaks = 30)
}
dev.off()

# Combine same patient
dat_combine <- data.frame(matrix(data = NA, ncol = dim(dat)[2]))
colnames(dat_combine) <- colnames(dat)
patient_id <- sapply(dat$slide_id, substr, 1, 12, simplify = T)
pID <- unique(patient_id) #628 slides, 390 patients
for (i in 1:length(pID)){
  thisPatient <- which(patient_id == pID[i])
  dat_combine[i, colnames] <- colMeans(dat[thisPatient, colnames], na.rm = T)
  dat_combine[i, 1:8] <- dat[thisPatient[1], 1:8]
}

# Set group = tumor_percent < 0.7, and only keep the first tissue patch
#group <- which(dat_combine$tumor_percent <= 0.5)
group <- 1:dim(dat_combine)[1]

# Univariate cox model
for(i in 1:length(colnames)){
  var = colnames[i]
  cox = coxph(Surv(time, status==1) ~ dat_combine[group, var], data = dat_combine[group, ])
  unicox_df[i,'var'] = var
  unicox_df[i,2:5] = summary(cox)$coef[c(2:5)]
  #sf <- survfit(surv_TCGA[group, ] ~ dat_combine_TCGA[group, var] >= median(dat_combine_TCGA[group, var], na.rm = T), 
  #        data = dat_combine_TCGA[group, ])
  #plot(sf, col = c("green", "red"), main = var, mark = 3)
  #cat(var, "\n")
  #print(survdiff(surv_TCGA[group, ] ~ dat_combine_TCGA[group, var] >= median(dat_combine_TCGA[group, var], na.rm = T), 
  #             data = dat_combine_TCGA[group, ]))
}
unicox_df

# Sort by p-value and identify features with p_value less than 0.05
#unicox_df = unicox_df[with(unicox_df, order(pv)),]
features = unicox_df$var
sig_features = features[with(unicox_df, pv<0.05)] 
hist(unicox_df$pv[which(!is.nan(unicox_df$pv))], breaks=seq(0,1,.05), labels=T) 

# Write to Excel file
write.csv(unicox_df, file = "univariate_patient_TCGA.csv", row.names = F)
