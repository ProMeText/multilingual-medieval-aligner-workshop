from bertalign import Bertalign



latin = "Suus deuotus Fr’ Aegidius Romanus Ordinis Fratrum Eremitarum S Augustini, cum recommendatione seipsum, et ad omnia famulatum. Clamat Politicorum sententia, omnes principatus non esse aequaliter diuturnos, nec aequali periodo singula regimina mensurari."

spanish = "El su deuoto fray Gil romano dela or dende sant Agustin: con muy humildosa recomendacion: assi mesmo para todo su seruicio. La sentencia de las politicas que quiere dezir sciencia de gouernamiento de las çibdades: dize assi. Que todos los principes no son ygualmente duraderos: ni tienen mesura todos los gouernamientos singulares por ygual mesura de tienpo."


aligner = Bertalign(latin, spanish)
aligner.align_sents()
aligner.print_sents()
