import matplotlib.pyplot as plt
import pymongo
import json
import pandas as pd
import numpy as np


if __name__ == '__main__':
    with open('./secrets.json') as sjson:
        secrets = json.load(sjson)

    with open('./check_fmo.out', 'r') as f:
        fmo_data = {l.split()[-2].replace('sciimg', 'scimrefdiffimg'): [float(l.split()[-5]),
                                                                        float(l.split()[-8])]
                    for l in f.readlines()[9:]}
    # print(fmo_data)
    # input()

    fig = plt.figure(figsize=(5, 4.85))
    fig.tight_layout()
    fig.subplots_adjust(top=0.911, bottom=0.097, left=0.137, right=0.978, hspace=0.195, wspace=0.198)
    ax1 = fig.add_subplot(111)
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)

    client = pymongo.MongoClient(host=secrets['deep_asteroids_mongodb']['host'],
                                 port=secrets['deep_asteroids_mongodb']['port'])

    db = client['deep-asteroids']
    db.authenticate(name=secrets['deep_asteroids_mongodb']['user'],
                    password=secrets['deep_asteroids_mongodb']['pwd'])

    streakids = ('strkid6433375921150001_pid643337592115',
                 'strkid6433978705150001_pid643397870515',
                 'strkid6692218501150002_pid669221850115',
                 'strkid6692380552150001_pid669238055215',
                 'strkid6692577715150001_pid669257771515',
                 'strkid6712424141150001_pid671242414115',
                 'strkid6722254623150002_pid672225462315',
                 'strkid6722282023150002_pid672228202315',
                 'strkid6722319523150001_pid672231952315',
                 'strkid6722328723150002_pid672232872315',
                 'strkid6724059045150002_pid672405904515',
                 'strkid6724259333150001_pid672425933315',
                 'strkid6731717262150001_pid673171726215',
                 'strkid6744154561150001_pid674415456115',
                 'strkid6751635944150002_pid675163594415',
                 'strkid6761231552150002_pid676123155215',
                 'strkid6771328206150006_pid677132820615',
                 'strkid6771881753150002_pid677188175315',
                 'strkid6771927635150001_pid677192763515',
                 'strkid6771936846150001_pid677193684615',
                 'strkid6772127835150001_pid677212783515',
                 'strkid6772602532150001_pid677260253215',
                 'strkid6772648832150001_pid677264883215',
                 'strkid6773168750150002_pid677316875015',
                 'strkid6773584436150002_pid677358443615',
                 'strkid6773941010150003_pid677394101015',
                 'strkid6782698661150001_pid678269866115',
                 'strkid6782707761150001_pid678270776115',
                 'strkid6801658719150001_pid680165871915',
                 'strkid6801721819150001_pid680172181915',
                 'strkid6801898458150002_pid680189845815',
                 'strkid6802069416150003_pid680206941615',
                 'strkid6802246716150001_pid680224671615',
                 'strkid6802551960150002_pid680255196015',
                 'strkid6812092515150005_pid681209251515',
                 'strkid6812134309150001_pid681213430915',
                 'strkid6812252744150005_pid681225274415',
                 'strkid6822366559150004_pid682236655915',
                 'strkid6832470453150001_pid683247045315',
                 'strkid6832958453150001_pid683295845315',
                 'strkid6833120301150001_pid683312030115',
                 'strkid6833998326150001_pid683399832615',
                 'strkid6834053440150002_pid683405344015',
                 'strkid6843886836150002_pid684388683615',
                 'strkid6844488441150001_pid684448844115',
                 'strkid6845067841150003_pid684506784115',
                 'strkid6864305427150002_pid686430542715',
                 'strkid6872582850150001_pid687258285015',
                 'strkid6934689916150002_pid693468991615',
                 'strkid6934698916150003_pid693469891615',
                 'strkid6935286125150001_pid693528612515',
                 'strkid7012623440150001_pid701262344015',
                 'strkid7012908142150001_pid701290814215',
                 'strkid7013055725150001_pid701305572515',
                 'strkid7013134942150002_pid701313494215',
                 'strkid7013316358150001_pid701331635815',
                 'strkid7013637941150004_pid701363794115',
                 'strkid7022030924150001_pid702203092415',
                 'strkid7022067916150002_pid702206791615',
                 'strkid7022739332150001_pid702273933215',
                 'strkid7032710936150002_pid703271093615',
                 'strkid7075061623150001_pid707506162315',
                 'strkid7083295750150001_pid708329575015',
                 'strkid7112275934150001_pid711227593415',
                 'strkid7112285034150001_pid711228503415',
                 'strkid7112490118150003_pid711249011815',
                 'strkid7112494612150004_pid711249461215',
                 'strkid7114642147150007_pid711464214715',
                 'strkid7122533735150001_pid712253373515',
                 'strkid7123248937150001_pid712324893715',
                 'strkid7141061558150002_pid714106155815',
                 'strkid7141116458150001_pid714111645815',
                 'strkid7141212358150004_pid714121235815',
                 'strkid7141349458150002_pid714134945815',
                 'strkid7141792541150007_pid714179254115',
                 # 'strkid7141820041150009_pid714182004115',
                 'strkid7153782624150002_pid715378262415',
                 'strkid7153800924150003_pid715380092415',
                 'strkid7153810024150002_pid715381002415',
                 'strkid7154458741150005_pid715445874115',
                 'strkid7154467941150003_pid715446794115',
                 'strkid7154477041150004_pid715447704115',
                 'strkid7154504541150003_pid715450454115',
                 'strkid7154513641150006_pid715451364115',
                 'strkid7154522841150002_pid715452284115',
                 'strkid7154577641150004_pid715457764115',
                 'strkid7154614141150002_pid715461414115',
                 'strkid7154659841150002_pid715465984115',
                 'strkid7201693443150001_pid720169344315',
                 'strkid7201757440150002_pid720175744015',
                 'strkid7201766540150002_pid720176654015',
                 'strkid7201771040150002_pid720177104015',
                 'strkid7201775640150002_pid720177564015',
                 'strkid7201780240150001_pid720178024015',
                 'strkid7201784840150002_pid720178484015',
                 'strkid7201789340150003_pid720178934015',
                 'strkid7201812140150002_pid720181214015',
                 'strkid7201830440150003_pid720183044015',
                 'strkid7201848740150003_pid720184874015',
                 'strkid7201880640150001_pid720188064015',
                 'strkid7201885340150001_pid720188534015',
                 'strkid7201903440150002_pid720190344015',
                 'strkid7201908140150004_pid720190814015',
                 'strkid7201921840150002_pid720192184015',
                 'strkid7201926340150001_pid720192634015',
                 'strkid7201935540150002_pid720193554015',
                 'strkid7201944640150001_pid720194464015',
                 'strkid7201962940150003_pid720196294015',
                 'strkid7201976740150003_pid720197674015',
                 'strkid7201985840150001_pid720198584015',
                 'strkid7201990340150002_pid720199034015',
                 'strkid7201999540150003_pid720199954015',
                 'strkid7202004140150003_pid720200414015',
                 'strkid7202008640150002_pid720200864015',
                 'strkid7202031440150003_pid720203144015',
                 'strkid7202036140150001_pid720203614015',
                 'strkid7202040640150003_pid720204064015',
                 'strkid7202049740150001_pid720204974015',
                 'strkid7202063440150001_pid720206344015',
                 'strkid7202072540150003_pid720207254015',
                 'strkid7202086340150002_pid720208634015',
                 'strkid7202095440150001_pid720209544015',
                 'strkid7202099940150002_pid720209994015',
                 'strkid7202118240150001_pid720211824015',
                 'strkid7202131940150002_pid720213194015',
                 'strkid7202136540150001_pid720213654015',
                 'strkid7202145740150002_pid720214574015',
                 'strkid7202168540150002_pid720216854015',
                 'strkid7202186840150002_pid720218684015',
                 'strkid7202191340150001_pid720219134015',
                 'strkid7202195940150001_pid720219594015',
                 'strkid7202200440150001_pid720220044015',
                 'strkid7202218740150001_pid720221874015',
                 'strkid7202223340150002_pid720222334015',
                 'strkid7202241640150002_pid720224164015',
                 'strkid7202246240150002_pid720224624015',
                 'strkid7202250840150001_pid720225084015',
                 'strkid7202269040150002_pid720226904015',
                 'strkid7202273640150001_pid720227364015',
                 'strkid7202287340150001_pid720228734015',
                 'strkid7202291840150001_pid720229184015',
                 'strkid7202301040150001_pid720230104015',
                 'strkid7202310140150002_pid720231014015',
                 'strkid7202319340150001_pid720231934015',
                 'strkid7202332940150002_pid720233294015',
                 'strkid7202337440150001_pid720233744015',
                 'strkid7202342140150001_pid720234214015',
                 'strkid7202364940150001_pid720236494015',
                 'strkid7202374040150001_pid720237404015',
                 'strkid7202383240150002_pid720238324015',
                 'strkid7202392340150001_pid720239234015',
                 'strkid7202401340150001_pid720240134015',
                 'strkid7202415140150001_pid720241514015',
                 'strkid7202428840150001_pid720242884015',
                 'strkid7202433440150002_pid720243344015',
                 'strkid7202437940150001_pid720243794015',
                 'strkid7202447140150001_pid720244714015',
                 'strkid7202451640150001_pid720245164015',
                 'strkid7202460740150001_pid720246074015',
                 'strkid7202469940150001_pid720246994015',
                 'strkid7202474540150001_pid720247454015',
                 'strkid7202483640150001_pid720248364015',
                 'strkid7202488140150001_pid720248814015',
                 'strkid7202492840150001_pid720249284015',
                 'strkid7202520140150001_pid720252014015',
                 'strkid7202529240150001_pid720252924015',
                 'strkid7202533940150001_pid720253394015',
                 'strkid7202543040150001_pid720254304015',
                 'strkid7202552140150001_pid720255214015',
                 'strkid7202771459150005_pid720277145915',
                 'strkid7202776059150005_pid720277605915',
                 'strkid7202780559150001_pid720278055915',
                 'strkid7202785159150004_pid720278515915',
                 'strkid7202789659150004_pid720278965915',
                 'strkid7202794359150003_pid720279435915',
                 'strkid7202798859150004_pid720279885915',
                 'strkid7202807959150003_pid720280795915',
                 'strkid7202812559150005_pid720281255915',
                 'strkid7202817159150004_pid720281715915',
                 'strkid7202821659150004_pid720282165915',
                 'strkid7211194427150001_pid721119442715',
                 'strkid7211518830150003_pid721151883015',
                 # 'strkid7215380663150001_pid721538066315',
                 'strkid7264724424150001_pid726472442415',
                 'strkid7274251740150001_pid727425174015',
                 'strkid7274256212150001_pid727425621215',
                 'strkid7274466515150001_pid727446651515',
                 'strkid7291579646150002_pid729157964615',
                 'strkid7291588735150001_pid729158873515',
                 'strkid7291692035150002_pid729169203515',
                 'strkid7331699915150001_pid733169991515',
                 'strkid7332361415150001_pid733236141515',
                 'strkid7362031407150002_pid736203140715',
                 'strkid7374170916150001_pid737417091615',
                 'strkid7374307119150001_pid737430711915',
                 # 'strkid7385107861150013_pid738510786115',
                 'strkid7392635635150001_pid739263563515',
                 'strkid7392676746150002_pid739267674615',
                 'strkid7394373448150001_pid739437344815',
                 'strkid7394387161150001_pid739438716115',
                 'strkid7394558319150001_pid739455831915',
                 'strkid7401755619150001_pid740175561915',
                 'strkid7401769330150001_pid740176933015',
                 'strkid7402187644150001_pid740218764415',
                 # 'strkid7402589524150003_pid740258952415',
                 'strkid7404437333150001_pid740443733315',
                 'strkid7404460633150001_pid740446063315',
                 'strkid7404841751150001_pid740484175115',
                 'strkid7404851151150001_pid740485115115',
                 'strkid7405192062150001_pid740519206215',
                 'strkid7405201662150001_pid740520166215',
                 'strkid7412887451150006_pid741288745115')

    c = list(db['deep-asteroids'].find({'_id': {'$in': streakids}}))

    cmap = {'vmag': [], 'rate': [], 'len': [], 'det': []}

    for s in c:
        plausible = ((s['rb_vgg6'] > 0.5) or (s['rb_resnet50'] > 0.5) or (s['rb_densenet121'] > 0.5)) and \
                    ((s['sl_vgg6'] > 0.5) or (s['sl_resnet50'] > 0.5) or (s['sl_densenet121'] > 0.5)) and \
                    ((s['kd_vgg6'] > 0.5) or (s['kd_resnet50'] > 0.5) or (s['kd_densenet121'] > 0.5))
        # if not plausible:
        #     print(s['_id'])
        fmo = fmo_data[s['pdiffimfilename']]
        # print(s['_id'], s['pdiffimfilename'], fmo)
        if fmo[1]*86400/3600 < 100:
            cmap['vmag'].append(fmo[0])
            cmap['rate'].append(fmo[1]*86400/3600)
            cmap['len'].append(fmo[1]*30)
            cmap['det'].append(1 if plausible else 0)

    df = pd.DataFrame(data=cmap)
    # print(df)

    df[df.det == 1].plot.scatter(x='rate', y='vmag', color='#00415a', marker='+', alpha=.0, s=90,
                                 label='Detections', ax=ax1)
    df[df.det == 0].plot.scatter(x='rate', y='vmag', color='#FF4136', marker='x', alpha=.7, s=60,
                                 label='Non-detections', ax=ax1)

    # df.plot.hexbin(x='rate', y='vmag', C='det', reduce_C_function=np.mean, gridsize=5, ax=ax2)

    ax1.set_xlabel('Motion rate (deg/day)')
    ax1.set_ylabel('V magnitude')

    ax2 = ax1.twiny()  # ax1 and ax2 share y-axis
    df[df.det == 1].plot.scatter(x='rate', y='vmag', color='#00415a', marker='+', alpha=.7, s=90,
                                 label='Detections', ax=ax2)
    df[df.det == 0].plot.scatter(x='rate', y='vmag', color='#FF4136', marker='x', alpha=.7, s=60,
                                 label='Non-detections', ax=ax2)
    ax2.set_xlabel('Length in ZTF image (pixel)')

    ax1.invert_yaxis()
    # ax2.invert_yaxis()

    fig.savefig(f'/Users/dmitryduev/_caltech/python/deep-asteroids/paper/neo_completeness.png', dpi=300)
    plt.show()
