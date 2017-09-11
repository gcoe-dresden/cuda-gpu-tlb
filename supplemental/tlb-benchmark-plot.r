#!/usr/bin/env Rscript

#
# Copyright (c) 2017 Tomas Karnagel and Matthias Werner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
#


args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

fpath<-sub("(.*\\/)([^.]+)(\\.[[:alnum:]]+$)", "\\1", args[1])
fname<-sub("(.*\\/)([^.]+)(\\.[[:alnum:]]+$)", "\\2", args[1])
cat("Plotting to ",fpath,fname, ".\n");
cairo_pdf(paste0(fpath,"plot-",fname,".pdf"), pointsize=9, width=150*0.03937, height=100*0.03937)

res <- readLines(args[1])
mylabels <- unlist(strsplit(res[2], split=","))
mydata <- read.csv(args[1], header=FALSE, comment.char = "#")

plot(0,0, 
	ylim=c(min(mydata[, -1], na.rm=TRUE)*0.9, max(mydata[, -1], na.rm=TRUE)*1.1), 
	xlim=c(min(mydata[, 1]), max(mydata[, 1])), 
	main="", ylab="cycles", xlab="traversed data size (MB)", type="l",  las=1);

grid (NULL,NULL, lty = 6, col = "cornsilk2")

for (column in 2:length(mydata)-1){
	x <- mydata[, 1];
	y <- mydata[, column];
	miss <- !is.na(y)
	lines(x[miss],y[miss], type="l", col=column-1, lty=1, lwd=1);
}
myLegend <- paste0(mylabels[2:(length(mydata)-1)],c(rep("KB stride ", length(mydata)-2)));

legend("top", horiz=TRUE, legend=myLegend, col=c(2:(length(mydata)-1))-1, lty=1);

dev.off()
