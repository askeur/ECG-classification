# Définir le style CSS personnalisé
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
         [data-testid=stSidebar] {
          background-color: #c9621e;


 
       }
        footer {visibility: hidden;}
        /* Changer la couleur de fond */
        body {
            font-family: "TT Norms Pro";
        }
        /* Changer la couleur du gros titre */
        h1 {
            color: #db5e0d;
        }
        /* Changer la couleur du texte */
        h5 {
            color: #ede8e8;
            font-size: 16px;
            font-family: "TT Norms Pro";
        }
        /* Changer la couleur du texte dans les paragraphes */
        p {
            color: #ede8e8;
        }
        /* Changer la couleur et la police des titres de niveau 2, 3 et 4 */
        h2, h3, h4 {
            font-family: "TT Norms Pro" !important;
            color: #ede8e8 !important; /*#db5e0d*/
        }
        /* Changer le style des puces dans les listes non ordonnées */
        ul {
            list-style-type: circle; /* Changer le type de puce ici (par exemple, square, circle, disc) */
            color: #bfbf99;
        }
        /* Changer le style des puces pour un élément spécifique de la liste */
        li {
            list-style-type: circle; /* Changer le type de puce ici */
            color: #bfbf99;
        }
        /* Utilisation de la police Baskerville pour les éléments spécifiques ayant la classe "example-class" */
        .example-class {
            font-family: "TT Norms Pro";
        }
        /* Personnaliser le style du pied de page */
        footer:after {
            content: 'DataScientest - FEVRIER 23 -' ;
            visibility: visible;
            display: block;
            position: relative;
            color: white; /* Changer la couleur du texte en blanc */
            padding: 5px;
            top: 2px;
        }
    </style>
 
"""
