import re


def split(string:str) -> list:
    # On va utiliser des subordonnant comme séparateurs pour aller au niveau du syntagme
    separator = r"[,;!?.:?¿]|( cum |donde| con | [Qq]ue | ut )"
    splits = re.split(separator, string)
    return [split for split in splits if split]


if __name__ == '__main__':
    string = "El su deuoto fray Gil romano de la orden de sant Agustin: " \
             "con muy humildosa recomendacion: assi mesmo para todo su seruicio. " \
             "La sentencia de las politicas que quiere dezir sciencia de gouernamiento " \
             "de las çibdades: dize assi. Que todos los principes no son ygualmente duraderos: " \
             "ni tienen mesura todos los gouernamientos singulares por ygual mesura de tienpo." \
             " Ca algunos gouernamientos son mesurados por vn año e otros por vida de un ombre. " \
             "Otros por heredamiento ⁊ por sucession de fiios que son iuzgados por las cosas naturales: " \
             "muestran ⁊ dizen: que ninguna co sa puede ser perpetua aqui en la tierra. Aquel que mucho " \
             "dessea quel su principado sea perpetuado en si ⁊ enlos fijos que vienen enpos del: " \
             "deue afincadamen te estudiar quel su gouernamiento sea natural: por que nunca puede " \
             "ser ninguno gouernador natural si siempre quiere go uernar con passion ⁊ con voluntad. " \
             "Mas aquel que es gouernador de iusticia no deue ordenar ni mandar ninguna cosa sin razon ⁊ sin ley. " \
             "Ca segund que dize el philosopho: assi como es naturalmente sieruo aquel que es fuerte en el " \
             "cuerpo ⁊ enlas virtudes corporales: ⁊ fallesçe enel entendimiento: assi aquel que es virtuoso ⁊ " \
             "poderoso enel entendimiento: naturalmente es señor por gouernamiento ⁊ por sabiduria: " \
             "que es razon derecha en to das las cosas que ha de fazer. " \
             "Por la qual razon si la vuestra gloriosa nobleza muy enamorosamente mando: " \
             "que yo conpusiesse vn libro de doctrina ⁊ gouernamiento segund razon ⁊ segund " \
             "ley pudiessedes naturalmente gouernar vuestro reyno: assi comomanifiestamente paresçe: " \
             "esta peticion no yi no por ombre: mas vins por dios Paresçe que dios que es todo poderoso " \
             "en cuya vestidura esta escripto. Señor de  los señores: rey de los reyes que ouo cuydado dela " \
             "santa casa vuestra: quando inclino la vuestra limpia ⁊ honrrada mançebia: que sigua las carreras" \
             " de vuestros padres ⁊ de vuestros predecessores donde venides Enlos quales siempre ha ⁊ ouo muy" \
             " acabadamente ⁊ muy complidamente zelo de fe ⁊ de religion christiana. El qual zelo ⁊ amor de fe " \
             "desseo sienpre guardar las reglas muy iustas: no por passion ni por voluntad: mas por ley ⁊ " \
             "entendimiento. Pues assi es por la vuestra peticion muy loada ⁊ muy honesta: la qual resçibo " \
             "por mandamiento. E avn por que an esto me induze el bien dela gente ⁊ el bien comun que es mas " \
             "diuinal que ningund bien especial ni personal. Por ende fuy mouido sin ninguna escusacion con el " \
             "ayuda de dios de començar con buena voluntad esta obra assi como la vuestra nobleza alta demando."

    latin = "Suus deuotus Fr’ Aegidius Romanus Ordinis Fratrum Eremitarum S Augustini, " \
            "cum recommendatione seipsum, et ad omnia famulatum. Clamat Politicorum sententia," \
            " omnes principatus non esse aequaliter diuturnos, nec aequali periodo singula regimina " \
            "mensurari. sed aliqua sunt annualia, aliqua ad vitam, aliqua vero per haereditatem et " \
            "successionem in filiis, quae quodammodo perpetua iudicantur. Cum igitur, nullum violentum" \
            " esse perpetuum, fere omnia naturalia protestentur, qui in se, et in suis posterioribus" \
            " filiis suum principatuum perpetuari desiderat, summopere studere debet, ut sit suum " \
            "regimen naturale. Nunquam autem naturalis quis rector efficitur, si passione, aut voluntate" \
            " cupiat principari, sed si custos iusti existens absque ratione, et lege nihil statuat " \
            "imperandum. Nam (ut testatur Philosophus) sicut est naturaliter seruus, qui pollens " \
            "viribus, deficit intellectu: sic vigens mentis industria, et regitiua prudentia," \
            " naturaliter dominatur. Quare si vestra generositas gloriosa me amicabiliter requisiuit," \
            " ut de eruditione Principum, siue de regimine Regum quendam librum componerem, " \
            "quatenus gubernatione regni secundum rationem, et legem diligentius circumspecta" \
            " polleretis regimine naturali, ut apparet ad liquidum, non instinctu humano, " \
            "sed potius diuino, huiusmodi petitio postulatur. Videtur enim Deus omnipotens," \
            " in cuius foemore scribitur, dominus dominantium et rex regum, vestrae domus" \
            " sanctissimae curam gerere specialem: cum vestram pudicam, ac venerabilem " \
            "infantiam inclinauit, ut sequens suorum patrum, ac praedecessorum vestigia," \
            " in quibus peramplius et perfectius viget et viguit zelus fidei et religio" \
            " christiana, non passione et voluntate, sed lege et intellectu regulas " \
            "regni iustissimas cupiat praeseruare. Hac igitur requisitione laudabili et" \
            " honesta, quam mihi reputo in praeceptum, necnon et suffragante bono gentis " \
            "et communi, quod est diuinius quam bonum aliquod singulare, irrecusabiliter " \
            "inclinatus (auxiliante altissimo) delectabiliter opus aggrediar, ut vestra " \
            "reuerenda nobilitas requisiuit. "

    print(split(string))
    print(split(latin))