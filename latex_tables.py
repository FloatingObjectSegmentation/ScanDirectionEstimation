import numpy as np

def predictions_to_latex_tables(swathspan, predictions, names):
    global name, i, x, format

    # PREDICTIONS TO LATEX TABLES
    mainlines = ''
    avghoughbynames = []
    stdhoughbynames = []
    avgderivbynames = []
    stdderivbynames = []
    avgairplanebynames = []
    stdairplanebynames = []
    for name in names:

        if name not in predictions.keys():
            break

        swathpreds = []
        derivpreds = []
        houghpreds = []

        for pred in predictions[name]:

            if pred.airplanepred_swath != []:
                swathpreds.append(pred.airplanepred_swath)
            if pred.derivpreds != []:
                derivpreds.append(pred.derivpreds)
            if pred.houghpreds != []:
                houghpreds.append(pred.houghpreds)

        avgpred = np.average(np.array(swathpreds))
        stddevpred = np.std(np.array(swathpreds))

        avgderiv = []
        stdderiv = []
        for i in range(len(swathspan)):
            x = np.average(np.array([p[i] for p in derivpreds]))
            y = np.std(np.array([p[i] for p in derivpreds]))
            avgderiv.append(x)
            stdderiv.append(y)

        avghough = []
        stdhough = []
        for i in range(len(swathspan)):
            x = np.average(np.array([p[i] for p in houghpreds]))
            y = np.std(np.array([p[i] for p in houghpreds]))
            avghough.append(x)
            stdhough.append(y)

        def format(num):
            return '{0:.2f}'.format(num)

        a = name.split('_')

        avgairplanebynames.append(avgpred)
        stdairplanebynames.append(stddevpred)
        avghoughbynames.append(np.min(np.array(avghough)))
        stdhoughbynames.append(np.min(np.array(stdhough)))
        avgderivbynames.append(np.min(np.array(avgderiv)))
        stdderivbynames.append(np.min(np.array(stdderiv)))

        # #main table
        # line = a[0] + '\\_' + a[1] + ' & '
        # line += format(avgpred) + ' & '
        # line += format(stddevpred) + ' & '
        # line += format(np.min(np.array(avghough))) + ' & '
        # line += format(np.min(np.array(stdhough))) + ' & '
        # line += format(np.min(np.array(avgderiv))) + ' & '
        # line += format(np.min(np.array(stdderiv))) + '\\\\ \n'
        # line += '\\hline'
        # print(line)

        # hough table by bmpsize
        # line = a[0] + '\\_' + a[1] + ' & '
        # for i in range(len(avghough)):
        #     line += format(avghough[i]) + ' & '
        # line = line[:-3] + '\\\\ \n'
        # line += '\\hline'
        # print(line)

        # deriv table by bmpsize
        line = a[0] + '\\_' + a[1] + ' & '
        for i in range(len(avgderiv)):
            line += format(avgderiv[i]) + ' & '
        line = line[:-3] + '\\\\ \n'
        line += '\\hline'
        print(line)

    # aggregate by names
    def avgbynames(array):
        return '\\textbf{' + format(np.average(np.array(array))) + "}"

    print('AVG BY NAMES')
    line = " & "
    line += avgbynames(avgairplanebynames) + " & "
    line += avgbynames(stdairplanebynames) + " & "
    line += avgbynames(avghoughbynames) + " & "
    line += avgbynames(stdhoughbynames) + " & "
    line += avgbynames(avgderivbynames) + " & "
    line += avgbynames(stdderivbynames) + "\\\\ \n"
    print(line)
